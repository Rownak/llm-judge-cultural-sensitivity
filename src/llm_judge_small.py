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

Judge prompt versions (--prompt-version)
-----------------------------------------
  0   : Simple  (no rubric, cultural sensitivity & emotional appropriateness)
  0.1 : Simple with locale-specific language
  0.2 : Simple with positional bias instruction
  0.3 : Simple with locale-specific language and positional bias instruction
  1   : Medium  (rubric v1, 7 dimensions, simplified flaws)     [default]
  2   : Advanced (rubric v2, applicability checks, score anchors)

Limitations
-----------
  SmolLM3-3B training languages: English, French, Spanish, Italian,
  Portuguese, Chinese, Arabic, Russian.
  Bangla (bn-BD) is NOT supported — judge quality on bn-BD rows will
  be lower than for API models.

Usage
-----
  python src/llm_judge_small.py                              # SmolLM3-3B + prompt version 1
  python src/llm_judge_small.py --prompt-version 0.3         # SmolLM3-3B + version 0.3
  python src/llm_judge_small.py --model HuggingFaceTB/SmolLM3-3B

  # 8 GB VRAM: add to .env  →  HF_LOAD_IN_8BIT=1
  # then run normally — 8-bit quantisation is applied automatically.

From notebook:
  from src.llm_judge_small import run_pairwise_judge_small
  results = run_pairwise_judge_small("data/test_set_subtle_synthetic_prompts.csv")
  results = run_pairwise_judge_small(
      "data/test_set_subtle_synthetic_prompts.csv",
      model="HuggingFaceTB/SmolLM3-3B",
      prompt_version="0.3",
  )
"""

import csv
import importlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Dynamic judge prompt loader (YAML-backed, same as llm_judge.py)
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).parent.parent / "prompts" / "judge_prompts.yaml"


@lru_cache(maxsize=None)
def _load_yaml_config() -> dict:
    """Load and cache the judge prompts YAML config."""
    with _YAML_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_judge_prompt(judge_version: str = "0", locale: str = "en-US"):
    """
    Load judge prompt from YAML config or rubric-based Python module.

    Parameters
    ----------
    judge_version : str
        Prompt version ("0", "0.1", "0.2", "0.3", "1", "2")
    locale : str
        Locale code (e.g., "en-US", "es-MX"). Used for locale-aware versions (0.1, 0.3).

    Returns
    -------
    tuple of (build_pairwise_prompt function, RUBRIC_VERSION string)
    """
    config = _load_yaml_config()

    if judge_version not in config:
        available = sorted(config.keys())
        raise ValueError(
            f"Unknown prompt version '{judge_version}'. Available: {available}"
        )

    entry = config[judge_version]
    rubric_version = entry["rubric_version"]

    # Rubric-based versions (v1, v2) delegate to their Python modules
    if entry.get("type") == "rubric-based":
        module_name = f"judge_prompt_v{judge_version.replace('.', '_')}"
        prompts_dir = str(Path(__file__).parent.parent / "prompts")
        if prompts_dir not in sys.path:
            sys.path.insert(0, prompts_dir)
        mod = importlib.import_module(module_name)
        return mod.build_pairwise_prompt, mod.RUBRIC_VERSION

    # Template-based versions (v0, v0.1, v0.2, v0.3)
    if entry.get("locale_aware"):
        # Pick template by language detected from locale
        lang = "es" if locale.startswith("es") else "en"
        templates = entry.get("templates", {})
        if lang not in templates:
            available_langs = list(templates.keys())
            raise ValueError(
                f"Prompt version '{judge_version}' has no template for locale '{locale}' "
                f"(language '{lang}'). Available: {available_langs}"
            )
        raw_template = templates[lang]
    else:
        raw_template = entry.get("template")
        if not raw_template:
            raise ValueError(
                f"Prompt version '{judge_version}' has no template defined"
            )

    def build_pairwise_prompt(
        locale_: str, prompt_text: str, response_a: str, response_b: str
    ) -> str:
        """Render the prompt template with the given inputs."""
        return raw_template.format(
            locale=locale_,
            prompt_text=prompt_text,
            response_a=response_a,
            response_b=response_b,
        ).strip()

    return build_pairwise_prompt, rubric_version


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
    prompt_version: str = "0",
    output_path: str | None = None,
) -> list[dict]:
    """
    Run pairwise judge on all rows in a CSV file using a local HF model.

    Parameters
    ----------
    csv_path        : path to test_set.csv (or dev_set.csv)
    model           : HuggingFace model repo ID (default: HuggingFaceTB/SmolLM3-3B)
    prompt_version  : prompt version ("0","0.1","0.2","0.3", "1", "2"); default "0"
    output_path     : output JSON path; auto-derived when None:
                      results/judge_{dataset_stem}_{model_slug}_v{prompt_version}.json

    Returns
    -------
    list[dict] — results for each row (for notebook display)
    """
    load_dotenv()

    build_pairwise_prompt_fn, rubric_version = _load_judge_prompt(prompt_version)

    # Load model once — kept in memory for the full batch
    tokenizer, hf_model = _load_model(model)

    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    print(f"Prompt Version: {prompt_version}")
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
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_stem = Path(csv_path).stem
        model_slug = model.replace("/", "-")
        output_path = output_dir / f"judge_{dataset_stem}_{model_slug}_v{prompt_version}.json"
    else:
        output_path = Path(output_path)

    output_data = {
        "prompt_version": prompt_version,
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
        "--prompt-version", default="0", choices=["0", "0.1", "0.2", "0.3", "1", "2"],
        help="Prompt version (default: 0)"
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
        prompt_version=args.prompt_version,
    )

    print()
    print("=" * 60)
    print(f"Complete: {len(results)} judgments recorded")
    print("=" * 60)
