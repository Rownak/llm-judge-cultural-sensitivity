"""
LLM-as-Judge: Pairwise evaluation using Claude or DeepSeek with structured output.

Loads any dev/test CSV, runs judge_prompt against each pair, saves results
to results/judge_{dataset_stem}_{model_slug}_v{rubric_version}.json.

Supported models
----------------
  anthropic : claude-haiku-4-5-20251001  (default)
              claude-sonnet-4-6
  deepseek  : deepseek-chat
              deepseek-reasoner

Prompt versions (--prompt-version)
-----------------------------------
  0   : Simple (no rubric, just cultural sensitivity & emotional appropriateness)
  0.1 : Simple with locale-specific language
  0.2 : Simple with positional bias instruction
  0.3 : Simple with locale-specific language and positional bias instruction
  1   : Medium (rubric v1 with 7 dimensions, simplified flaws)
  2   : Advanced (rubric v2 with applicability checks, score anchors)

Usage:
  python src/llm_judge.py                                    # Claude Haiku + version 1 (default)
  python src/llm_judge.py --model deepseek-chat               # DeepSeek Chat + version 1
  python src/llm_judge.py --prompt-version 0.3               # Claude Haiku + version 0.3
  python src/llm_judge.py --prompt-version 2 --model deepseek-chat

From notebook:
  from src.llm_judge import run_pairwise_judge
  results = run_pairwise_judge("data/test_set_subtle_synthetic_prompts.csv", model="claude-haiku-4-5-20251001")
  results = run_pairwise_judge("data/test_set_subtle_synthetic_prompts.csv", prompt_version="0.3", model="deepseek-chat")
"""

import csv
import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Dynamic judge prompt loader (YAML-backed)
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).parent.parent / "prompts" / "judge_prompts.yaml"


@lru_cache(maxsize=None)
def _load_yaml_config() -> dict:
    """Load and cache the judge prompts YAML config."""
    with _YAML_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_judge_prompt(judge_version: str = "1", locale: str = "en-US"):
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


build_pairwise_prompt, RUBRIC_VERSION = _load_judge_prompt(judge_version="1")


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
}

DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _provider(model: str) -> str:
    """Return 'anthropic' or 'deepseek' for a given model name."""
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in DEEPSEEK_MODELS:
        return "deepseek"
    raise ValueError(
        f"Unknown model '{model}'. "
        f"Supported: {sorted(ANTHROPIC_MODELS | DEEPSEEK_MODELS)}"
    )


def _build_client(model: str):
    """Build the appropriate API client based on model provider."""
    provider = _provider(model)

    if provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        return anthropic.Anthropic(api_key=api_key)

    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# Pydantic schema for structured output
# ---------------------------------------------------------------------------

class JudgeOutput(BaseModel):
    """Pairwise judgment output matching rubric v1 schema."""
    preferred: Literal["A", "B", "Tie"]
    confidence: Literal["high", "medium", "low"]
    rationale: str
    flaws_in_A: list[str]
    flaws_in_B: list[str]


# Shared tool / function definition (used by both providers)
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
# Provider-specific call functions
# ---------------------------------------------------------------------------

def _call_anthropic(client, model: str, judge_prompt: str) -> JudgeOutput:
    """Call Anthropic API with tool_use to force structured output."""
    # Anthropic uses "input_schema" instead of "parameters"
    anthropic_tool = {
        "name": JUDGMENT_TOOL_SCHEMA["name"],
        "description": JUDGMENT_TOOL_SCHEMA["description"],
        "input_schema": JUDGMENT_TOOL_SCHEMA["parameters"],
    }

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[anthropic_tool],
        tool_choice={"type": "tool", "name": "record_judgment"},
        messages=[{"role": "user", "content": judge_prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return JudgeOutput(**block.input)

    raise RuntimeError("No tool_use block in Anthropic response")


def _call_deepseek(client, model: str, judge_prompt: str) -> JudgeOutput:
    """Call DeepSeek (OpenAI-compatible) API with function_call for structured output."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        tools=[{"type": "function", "function": JUDGMENT_TOOL_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "record_judgment"}},
        messages=[{"role": "user", "content": judge_prompt}]
    )

    tool_call = response.choices[0].message.tool_calls
    if not tool_call:
        raise RuntimeError("No tool_call in DeepSeek response")

    args = json.loads(tool_call[0].function.arguments)
    return JudgeOutput(**args)


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def run_judge(row: dict, client, model: str, build_pairwise_prompt_fn) -> dict:
    """
    Run judge on a single pair.

    Parameters
    ----------
    row                      : dict from test_set.csv (prompt_id, locale, prompt_text, response_A, response_B, ...)
    client                   : initialized API client (Anthropic or OpenAI)
    model                    : model name string
    build_pairwise_prompt_fn : the build_pairwise_prompt function from the selected judge prompt module

    Returns
    -------
    dict — input row merged with judge output fields
    """
    judge_prompt = build_pairwise_prompt_fn(
        row["locale"], row["prompt_text"], row["response_A"], row["response_B"]
    )

    provider = _provider(model)
    if provider == "anthropic":
        judge_output = _call_anthropic(client, model, judge_prompt)
    else:
        judge_output = _call_deepseek(client, model, judge_prompt)

    result = dict(row)
    result.update(judge_output.model_dump())
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_pairwise_judge(
    csv_path: str,
    model: str = DEFAULT_MODEL,
    prompt_version: str = "1",
    output_path: str | None = None,
) -> list[dict]:
    """
    Run pairwise judge on all rows in a CSV file.

    Parameters
    ----------
    csv_path        : path to test_set.csv (or dev_set.csv)
    model           : model name — selects provider automatically
    prompt_version  : prompt version ("0", "0.1", "0.2", "0.3", "1", "2"); defaults to "1"
    output_path     : output JSON path; defaults to
                      results/judge_{dataset_stem}_{model_slug}_v{prompt_version}.json

    Returns
    -------
    list[dict] — results for each row (for notebook display)
    """
    load_dotenv()
    client = _build_client(model)
    build_pairwise_prompt_fn, rubric_version = _load_judge_prompt(prompt_version)

    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    print(f"Prompt Version: {prompt_version}")
    print(f"Model        : {model}")
    print(f"Provider     : {_provider(model)}")
    print(f"Dataset      : {csv_path} ({len(rows)} rows)")
    print()

    results = []
    for i, row in enumerate(rows, 1):
        prompt_id = row.get("prompt_id", f"row_{i}")
        print(f"  [{i}/{len(rows)}] {prompt_id} ...", end=" ", flush=True)
        try:
            result = run_judge(row, client, model, build_pairwise_prompt_fn)
            results.append(result)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    # Build output path: output/judge_{dataset_stem}_{model_slug}_v{prompt_version}.json
    if output_path is None:
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_stem = Path(csv_path).stem   # e.g. "test_set_synthetic_prompts"
        model_slug = model.replace("/", "-")
        output_path = output_dir / f"judge_{dataset_stem}_{model_slug}_v{prompt_version}.json"
    else:
        output_path = Path(output_path)

    output_data = {
        "prompt_version": prompt_version,
        "rubric_version": rubric_version,
        "model": model,
        "n_evaluated": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Main script  (python src/llm_judge.py [--model MODEL])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run LLM-as-Judge pairwise evaluation")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prompt-version", default="1", choices=["0", "0.1", "0.2", "0.3", "1", "2"],
        help="Prompt version (default: 1)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input CSV (default: data/test_set_subtle_synthetic_prompts.csv)"
    )
    args = parser.parse_args()

    input_path = args.input or str(Path(__file__).parent.parent / "data" / "test_set_subtle_synthetic_prompts.csv")

    print("=" * 60)
    print("LLM-as-Judge: Pairwise Evaluation")
    print("=" * 60)

    results = run_pairwise_judge(input_path, model=args.model, prompt_version=args.prompt_version)

    print()
    print("=" * 60)
    print(f"Complete: {len(results)} judgments recorded")
    print("=" * 60)
