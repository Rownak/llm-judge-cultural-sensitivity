"""
LLM-as-Judge (Small): Pairwise evaluation using a local HuggingFace model.

Mirrors the interface of llm_judge.py but runs inference locally via
the `transformers` library instead of a cloud API.  No API key needed.

Supported models
----------------
  HuggingFaceTB/SmolLM3-3B  (default)

  Any causal LM with a chat template that accepts `xml_tools` will work.
  Pass the full HuggingFace repo ID as --model.

Memory requirements (SmolLM3-3B, ~6.15 GB weights)
----------------------------------------------------
  bfloat16 default   : ~8-9 GB VRAM
  8-bit quantised    : ~4-5 GB VRAM — set HF_LOAD_IN_8BIT=1 in .env
                       (requires:  pip install bitsandbytes)

  `device_map="auto"` handles single GPU, multi-GPU, and CPU fallback.

Judge prompt versions (--judge-version)
---------------------------------------
  v0 : Simple  (no rubric, cultural sensitivity & emotional appropriateness)
  v1 : Medium  (rubric v1, 7 dimensions, simplified flaws)     [default]
  v2 : Advanced (rubric v2, applicability checks, score anchors)

Limitations
-----------
  SmolLM3-3B training languages: English, French, Spanish, Italian,
  Portuguese, Chinese, Arabic, Russian.
  Bangla (bn-BD) is NOT supported — judge quality on bn-BD rows will
  be lower than for API models.

Usage
-----
  python src/llm_judge_small.py                              # SmolLM3-3B + v1
  python src/llm_judge_small.py --judge-version v2           # SmolLM3-3B + v2
  python src/llm_judge_small.py --model HuggingFaceTB/SmolLM3-3B

  # 8 GB VRAM: add to .env  →  HF_LOAD_IN_8BIT=1
  # then run normally — 8-bit quantisation is applied automatically.

From notebook:
  from src.llm_judge_small import run_pairwise_judge_small
  results = run_pairwise_judge_small("data/test_set_subtle_synthetic_prompts.csv")
  results = run_pairwise_judge_small(
      "data/test_set_subtle_synthetic_prompts.csv",
      model="HuggingFaceTB/SmolLM3-3B",
      judge_version="v2",
  )
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Dynamic judge prompt loader  (same helper as llm_judge.py)
# ---------------------------------------------------------------------------

def _load_judge_prompt(judge_version: str = "v0"):
    """
    Dynamically import the appropriate judge prompt module.

    Returns
    -------
    tuple of (build_pairwise_prompt function, RUBRIC_VERSION string)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))

    if judge_version == "v0":
        from judge_prompt_v0 import build_pairwise_prompt, RUBRIC_VERSION
    elif judge_version == "v1":
        from judge_prompt_v1 import build_pairwise_prompt, RUBRIC_VERSION
    elif judge_version == "v2":
        from judge_prompt_v2 import build_pairwise_prompt, RUBRIC_VERSION
    else:
        raise ValueError(f"Unknown judge version '{judge_version}'. Supported: v0, v1, v2")

    return build_pairwise_prompt, RUBRIC_VERSION


# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"


# ---------------------------------------------------------------------------
# Pydantic schema for structured output  (identical to llm_judge.py)
# ---------------------------------------------------------------------------

class JudgeOutput(BaseModel):
    """Pairwise judgment output matching rubric v1 schema."""
    preferred: Literal["A", "B", "Tie"]
    confidence: Literal["high", "medium", "low"]
    rationale: str
    flaws_in_A: list[str]
    flaws_in_B: list[str]


class JudgeParseError(Exception):
    """Raised when the model's output cannot be parsed into a JudgeOutput.

    Small local models occasionally emit malformed JSON or omit required
    fields.  The batch runner catches this and skips the row instead of
    aborting the whole run.
    """


# Shared tool schema — same shape as llm_judge.py so SmolLM3's xml_tools
# path receives the exact same JSON Schema the API models receive.
JUDGMENT_TOOL_SCHEMA = {
    "name": "record_judgment",
    "description": "Record the pairwise judgment results",
    "parameters": {
        "type": "object",
        "properties": {
            "preferred": {
                "type": "string",
                "enum": ["A", "B", "Tie"],
                "description": "Which response is preferred"
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence level of judgment"
            },
            "rationale": {
                "type": "string",
                "description": "Reasoning for the judgment (1-3 sentences)"
            },
            "flaws_in_A": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 'CRITERION_ID: flaw_label' strings"
            },
            "flaws_in_B": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 'CRITERION_ID: flaw_label' strings"
            }
        },
        "required": ["preferred", "confidence", "rationale", "flaws_in_A", "flaws_in_B"]
    }
}


# ---------------------------------------------------------------------------
# Model loading  (once per batch run, reused for every row)
# ---------------------------------------------------------------------------

def _load_model(model_id: str):
    """
    Load a local HuggingFace causal LM and its tokenizer.

    Reads HF_LOAD_IN_8BIT from the environment:
      - "1"  → load in 8-bit via bitsandbytes  (recommended for 8 GB VRAM)
      - anything else → bfloat16 default        (recommended for 16 GB+ VRAM)

    Returns
    -------
    (tokenizer, model) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_in_8bit = os.getenv("HF_LOAD_IN_8BIT", "0") == "1"

    print(f"Loading model  : {model_id}")
    if load_in_8bit:
        print("Quantisation   : 8-bit (HF_LOAD_IN_8BIT=1)")
    else:
        print("Quantisation   : none (bfloat16)")
    print("(first run downloads weights — this may take a few minutes)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if load_in_8bit:
        # Modern transformers requires BitsAndBytesConfig instead of load_in_8bit=True
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    return tokenizer, hf_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _strip_output_format_section(prompt: str) -> str:
    """
    Remove the trailing 'OUTPUT FORMAT' section from a judge prompt.

    The shared judge_prompt_v{0,1,2} modules append a JSON-schema block that
    tells the model 'Return ONLY valid JSON ...'.  That instruction competes
    with SmolLM3's xml_tools tool-calling format and confuses the small model.
    By stripping the section here, the model sees a clean problem statement
    and is free to follow the xml_tools <tool_call> protocol from its chat
    template instead.
    """
    marker = "OUTPUT FORMAT"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt
    # Walk back to the divider line above the marker (the row of ═ characters)
    cut = prompt.rfind("\n", 0, idx)
    if cut == -1:
        return prompt
    divider_start = prompt.rfind("\n", 0, cut)
    return prompt[:divider_start].rstrip() if divider_start != -1 else prompt[:cut].rstrip()


def _call_local(tokenizer, hf_model, judge_prompt: str) -> JudgeOutput:
    """
    Run one inference call against the local model using xml_tools.

    SmolLM3-3B's chat template natively supports xml_tools — it emits:
        <tool_call>{"name": "record_judgment", "arguments": {...}}</tool_call>

    We strip the redundant 'Return ONLY valid JSON' block from the judge
    prompt so the model receives a single, coherent instruction: call the
    record_judgment tool.
    """
    cleaned_prompt = _strip_output_format_section(judge_prompt)
    messages = [{"role": "user", "content": cleaned_prompt}]
    tools = [JUDGMENT_TOOL_SCHEMA]

    inputs = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,   # disable <think>…</think> trace for speed
        xml_tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(hf_model.device)

    input_len = inputs["input_ids"].shape[-1]

    outputs = hf_model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )

    # Decode only the newly generated tokens (skip the prompt)
    decoded = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )
    # Extract the JSON payload — SmolLM3 emits one of three shapes:
    #   1. <tool_call>{...}</tool_call>   (fully wrapped, ideal)
    #   2. <tool_call>{...}               (opening tag, no close)
    #   3. {...}                          (bare JSON, no wrapper)
    m = re.search(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>)?", decoded, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        # Fall back to the first balanced {...} block in the raw output
        first = decoded.find("{")
        last = decoded.rfind("}")
        if first == -1 or last <= first:
            raise JudgeParseError(
                f"No JSON payload found in model output.\n"
                f"Raw output (first 500 chars):\n{decoded[:500]}"
            )
        json_str = decoded[first:last + 1]

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise JudgeParseError(
            f"Malformed JSON in model output ({e}).\n"
            f"Raw output (first 500 chars):\n{decoded[:500]}"
        ) from e

    # SmolLM3 wraps args under {"name": ..., "arguments": {...}}
    args = payload.get("arguments", payload)
    try:
        return JudgeOutput(**args)
    except (TypeError, ValueError) as e:
        # Pydantic ValidationError is a ValueError subclass — caught here too.
        raise JudgeParseError(
            f"Model output did not match JudgeOutput schema ({e}).\n"
            f"Parsed args: {args}"
        ) from e


# ---------------------------------------------------------------------------
# Per-row judge runner
# ---------------------------------------------------------------------------

def run_judge_small(row: dict, tokenizer, hf_model, build_pairwise_prompt_fn) -> dict:
    """
    Run judge on a single CSV row.

    Parameters
    ----------
    row                      : dict with keys prompt_id, locale, prompt_text,
                               response_A, response_B (plus any extras)
    tokenizer                : HuggingFace tokenizer (loaded once by caller)
    hf_model                 : HuggingFace model     (loaded once by caller)
    build_pairwise_prompt_fn : prompt builder from the selected judge version

    Returns
    -------
    dict — input row merged with judge output fields
    """
    judge_prompt = build_pairwise_prompt_fn(
        row["locale"], row["prompt_text"], row["response_A"], row["response_B"]
    )
    judge_output = _call_local(tokenizer, hf_model, judge_prompt)

    result = dict(row)
    result.update(judge_output.model_dump())
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_pairwise_judge_small(
    csv_path: str,
    model: str = DEFAULT_MODEL,
    judge_version: str = "v0",
    output_path: str | None = None,
) -> list[dict]:
    """
    Run pairwise judge on all rows in a CSV file using a local HF model.

    Parameters
    ----------
    csv_path       : path to test_set.csv (or dev_set.csv)
    model          : HuggingFace model repo ID (default: HuggingFaceTB/SmolLM3-3B)
    judge_version  : judge prompt version ("v0", "v1", "v2"); default "v1"
    output_path    : output JSON path; auto-derived when None:
                     results/judge_{dataset_stem}_{model_slug}_v{judge_version}.json

    Returns
    -------
    list[dict] — results for each row (for notebook display)
    """
    load_dotenv()

    build_pairwise_prompt_fn, rubric_version = _load_judge_prompt(judge_version)

    # Load model once — kept in memory for the full batch
    tokenizer, hf_model = _load_model(model)

    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    print(f"Judge Version: {judge_version}")
    print(f"Model        : {model}")
    print(f"Dataset      : {csv_path} ({len(rows)} rows)")
    print()

    results = []
    skipped = []
    for i, row in enumerate(rows, 1):
        prompt_id = row.get("prompt_id", f"row_{i}")
        print(f"  [{i}/{len(rows)}] {prompt_id} ...", end=" ", flush=True)
        try:
            result = run_judge_small(row, tokenizer, hf_model, build_pairwise_prompt_fn)
            results.append(result)
            print("done")
        except JudgeParseError as e:
            # Bad JSON from the small model — skip this row but keep going.
            skipped.append({"prompt_id": prompt_id, "reason": str(e).splitlines()[0]})
            print(f"SKIPPED (bad JSON): {str(e).splitlines()[0]}")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    if skipped:
        print()
        print(f"Skipped {len(skipped)} row(s) due to unparseable model output:")
        for s in skipped:
            print(f"  - {s['prompt_id']}: {s['reason']}")

    # Derive output path
    if output_path is None:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        dataset_stem = Path(csv_path).stem
        model_slug = model.replace("/", "-")
        output_path = results_dir / f"judge_{dataset_stem}_{model_slug}_v{judge_version}.json"
    else:
        output_path = Path(output_path)

    output_data = {
        "judge_version": judge_version,
        "rubric_version": rubric_version,
        "model": model,
        "n_evaluated": len(results),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge pairwise evaluation with a local HuggingFace model"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model repo ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--judge-version", default="v1", choices=["v0", "v1", "v2"],
        help="Judge prompt version (default: v1)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input CSV (default: data/test_set_subtle_synthetic_prompts.csv)"
    )
    args = parser.parse_args()

    input_path = (
        args.input
        or str(Path(__file__).parent.parent / "data" / "test_set_subtle_synthetic_prompts.csv")
    )

    print("=" * 60)
    print("LLM-as-Judge (Small): Pairwise Evaluation")
    print("=" * 60)

    results = run_pairwise_judge_small(
        input_path,
        model=args.model,
        judge_version=args.judge_version,
    )

    print()
    print("=" * 60)
    print(f"Complete: {len(results)} judgments recorded")
    print("=" * 60)
