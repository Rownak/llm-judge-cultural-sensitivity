"""
LLM-as-Judge: Pairwise evaluation using Claude or DeepSeek with structured output.

Loads test_set.csv, runs judge_prompt_v1 against each pair, saves results
to results/judge_results_{model_slug}_v{rubric_version}.json.

Supported models
----------------
  anthropic : claude-haiku-4-5-20251001  (default)
              claude-sonnet-4-6
  deepseek  : deepseek-chat
              deepseek-reasoner

Usage:
  python src/llm_judge.py                          # Claude Haiku (default)
  python src/llm_judge.py --model deepseek-chat    # DeepSeek Chat

From notebook:
  from src.llm_judge import run_pairwise_judge
  results = run_pairwise_judge("data/test_set.csv", model="claude-haiku-4-5-20251001")
  results = run_pairwise_judge("data/test_set.csv", model="deepseek-chat")
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Load judge prompt builder
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))
from judge_prompt_v1 import build_pairwise_prompt, RUBRIC_VERSION


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

def run_judge(row: dict, client, model: str) -> dict:
    """
    Run judge on a single pair.

    Parameters
    ----------
    row    : dict from test_set.csv (prompt_id, locale, prompt_text, response_A, response_B, ...)
    client : initialized API client (Anthropic or OpenAI)
    model  : model name string

    Returns
    -------
    dict — input row merged with judge output fields
    """
    judge_prompt = build_pairwise_prompt(
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
    output_path: str | None = None,
) -> list[dict]:
    """
    Run pairwise judge on all rows in a CSV file.

    Parameters
    ----------
    csv_path    : path to test_set.csv (or dev_set.csv)
    model       : model name — selects provider automatically
    output_path : output JSON path; defaults to
                  results/judge_results_{model_slug}_v{RUBRIC_VERSION}.json

    Returns
    -------
    list[dict] — results for each row (for notebook display)
    """
    load_dotenv()
    client = _build_client(model)

    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    print(f"Model    : {model}")
    print(f"Provider : {_provider(model)}")
    print(f"Dataset  : {csv_path} ({len(rows)} rows)")
    print()

    results = []
    for i, row in enumerate(rows, 1):
        prompt_id = row.get("prompt_id", f"row_{i}")
        print(f"  [{i}/{len(rows)}] {prompt_id} ...", end=" ", flush=True)
        try:
            result = run_judge(row, client, model)
            results.append(result)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    # Build output path: results/judge_results_claude-haiku-4-5_v1.0.json
    if output_path is None:
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        model_slug = model.replace("/", "-")
        output_path = results_dir / f"judge_results_{model_slug}_v{RUBRIC_VERSION}.json"
    else:
        output_path = Path(output_path)

    output_data = {
        "rubric_version": RUBRIC_VERSION,
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
        "--input", default=None,
        help="Path to input CSV (default: data/test_set.csv)"
    )
    args = parser.parse_args()

    input_path = args.input or str(Path(__file__).parent.parent / "data" / "test_set.csv")

    print("=" * 60)
    print("LLM-as-Judge: Pairwise Evaluation")
    print("=" * 60)

    results = run_pairwise_judge(input_path, model=args.model)

    print()
    print("=" * 60)
    print(f"Complete: {len(results)} judgments recorded")
    print("=" * 60)
