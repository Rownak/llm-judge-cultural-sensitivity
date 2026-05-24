"""
Judge prompt template for rubric v1 (judge_rubric_v1.json).

v1 field shape:
  - dimension["flaws"]          → list[str]
  - dimension["severity"]       → str  (dimension-level, not per-flaw)
  - dimension["locale_notes"]   → dict[locale, str]
  - pairwise_output_schema flaws_in_A/B → list[str]  "CS-1: register_too_casual"
  - No applicable_when / non_applicable_when
  - No score_anchors
  - No judge_instructions
  - No rubric_version in output
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Load rubric
# ---------------------------------------------------------------------------

RUBRIC_PATH = Path(__file__).parent.parent / "rubrics" / "judge_rubric_v1.json"

with RUBRIC_PATH.open(encoding="utf-8") as f:
    RUBRIC = json.load(f)

# Build lookup: id -> dimension dict
CRITERIA: dict[str, dict] = {d["id"]: d for d in RUBRIC["dimensions"]}

# Flaw taxonomy lookup: label -> metadata
FLAW_LOOKUP: dict[str, dict] = {f["label"]: f for f in RUBRIC["flaw_taxonomy"]}

RUBRIC_VERSION = RUBRIC["version"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_flaws_for_criterion(criterion_id: str) -> str:
    """Return a bullet list of flaw labels + descriptions for a criterion."""
    dim = CRITERIA[criterion_id]
    lines = []
    for label in dim["flaws"]:
        meta = FLAW_LOOKUP.get(label, {})
        desc = meta.get("description", "")
        severity = meta.get("severity", dim["severity"])
        lines.append(f"    - [{severity.upper()}] `{label}`: {desc}")
    return "\n".join(lines)


def _format_locale_note(criterion_id: str, locale: str) -> str:
    note = CRITERIA[criterion_id]["locale_notes"].get(locale, "No specific note for this locale.")
    return note


def _severity_table() -> str:
    sw = RUBRIC["severity_weighting"]
    return (
        f"  • CRITICAL : {sw['critical']}\n"
        f"  • MODERATE : {sw['moderate']}\n"
        f"  • MINOR    : {sw['minor']}"
    )


def _pointwise_scale() -> str:
    lines = []
    for entry in RUBRIC["pointwise_scale"]:
        score = entry["score"] if entry["score"] is not None else "N/A"
        lines.append(f"  {score} – {entry['label']}: {entry['meaning']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_pairwise_prompt(
    locale: str,
    prompt_text: str,
    response_a: str,
    response_b: str,
    active_criteria: list[str] | None = None,
) -> str:
    """
    Build a pairwise judge prompt for rubric v1.

    Parameters
    ----------
    locale          : one of "en-US", "bn-BD", "ja-JP"
    prompt_text     : the original user-facing prompt being evaluated
    response_a      : first AI response (labelled A)
    response_b      : second AI response (labelled B)
    active_criteria : subset of criterion IDs to evaluate; defaults to all 7
    """
    if active_criteria is None:
        active_criteria = [d["id"] for d in RUBRIC["dimensions"]]

    # --- criteria section -------------------------------------------------
    criteria_block_lines = []
    for cid in active_criteria:
        dim = CRITERIA[cid]
        criteria_block_lines.append(
            f"### {cid} · {dim['name']}  [severity: {dim['severity'].upper()}]\n"
            f"Definition: {dim['definition']}\n"
            f"Locale note ({locale}): {_format_locale_note(cid, locale)}\n"
            f"Flaws to watch for:\n{_format_flaws_for_criterion(cid)}"
        )
    criteria_block = "\n\n".join(criteria_block_lines)

    # --- output format section --------------------------------------------
    output_schema = RUBRIC["pairwise_output_schema"]
    output_format = (
        "Return ONLY valid JSON — no prose before or after — matching this schema:\n\n"
        "```json\n"
        "{\n"
        f'  "preferred": {json.dumps(output_schema["preferred"]["enum"])},\n'
        f'  "confidence": {json.dumps(output_schema["confidence"]["enum"])},\n'
        '  "rationale": "<one to three sentences citing specific criterion codes>",\n'
        '  "flaws_in_A": ["<criterion_id: flaw_label>", ...],\n'
        '  "flaws_in_B": ["<criterion_id: flaw_label>", ...]\n'
        "}\n"
        "```\n\n"
        "flaws_in_A / flaws_in_B format: \"CS-1: register_too_casual\"  "
        "(criterion id, colon, flaw label from the taxonomy above).\n"
        "Use an empty list [] if no flaws were found.\n"
        f"Tie policy: {RUBRIC['tie_policy']}"
    )

    # --- assemble ---------------------------------------------------------
    prompt = f"""You are a calibrated cultural-sensitivity and emotional-appropriateness judge.
Rubric version: {RUBRIC_VERSION}

═══════════════════════════════════════════════════════
LOCALE UNDER EVALUATION
═══════════════════════════════════════════════════════
{locale}

═══════════════════════════════════════════════════════
USER PROMPT BEING EVALUATED
═══════════════════════════════════════════════════════
{prompt_text}

═══════════════════════════════════════════════════════
RESPONSE A
═══════════════════════════════════════════════════════
{response_a}

═══════════════════════════════════════════════════════
RESPONSE B
═══════════════════════════════════════════════════════
{response_b}

═══════════════════════════════════════════════════════
EVALUATION CRITERIA
═══════════════════════════════════════════════════════
{criteria_block}

═══════════════════════════════════════════════════════
SEVERITY WEIGHTING GUIDE
═══════════════════════════════════════════════════════
{_severity_table()}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Compare Response A and Response B against every criterion listed above.
For each criterion:
  1. Check both responses for the listed flaws.
  2. Note the severity of any flaw found.
  3. Weigh flaws using the severity guide above.

Then choose the better response (A, B, or Tie) and explain why,
citing at least one criterion code in your rationale.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
{output_format}
"""
    return prompt.strip()


def build_pointwise_prompt(
    locale: str,
    prompt_text: str,
    response: str,
    response_label: str = "Response",
    active_criteria: list[str] | None = None,
) -> str:
    """
    Build a pointwise 1-5 scoring prompt for rubric v1.

    Parameters
    ----------
    locale          : one of "en-US", "bn-BD", "ja-JP"
    prompt_text     : the original user-facing prompt being evaluated
    response        : the single AI response to score
    response_label  : display label (e.g. "Response A")
    active_criteria : subset of criterion IDs to score; defaults to all 7
    """
    if active_criteria is None:
        active_criteria = [d["id"] for d in RUBRIC["dimensions"]]

    # --- criteria section -------------------------------------------------
    criteria_lines = []
    for cid in active_criteria:
        dim = CRITERIA[cid]
        criteria_lines.append(
            f"  {cid} · {dim['name']}\n"
            f"    Definition : {dim['definition']}\n"
            f"    Locale note: {_format_locale_note(cid, locale)}\n"
            f"    Flaws:\n{_format_flaws_for_criterion(cid)}"
        )
    criteria_block = "\n\n".join(criteria_lines)

    # --- score output section ---------------------------------------------
    score_fields = "\n".join(
        f'  "{cid}": {{"score": <1-5 or null>, "rationale": "<one sentence>"}}'
        for cid in active_criteria
    )
    output_format = (
        "Return ONLY valid JSON:\n\n"
        "```json\n"
        "{\n"
        f"{score_fields}\n"
        "}\n"
        "```\n\n"
        "Use null for N/A (criterion not applicable to this prompt).\n"
        f"Score meanings:\n{_pointwise_scale()}"
    )

    prompt = f"""You are a calibrated cultural-sensitivity and emotional-appropriateness judge.
Rubric version: {RUBRIC_VERSION}

═══════════════════════════════════════════════════════
LOCALE UNDER EVALUATION
═══════════════════════════════════════════════════════
{locale}

═══════════════════════════════════════════════════════
USER PROMPT BEING EVALUATED
═══════════════════════════════════════════════════════
{prompt_text}

═══════════════════════════════════════════════════════
{response_label.upper()}
═══════════════════════════════════════════════════════
{response}

═══════════════════════════════════════════════════════
EVALUATION CRITERIA
═══════════════════════════════════════════════════════
{criteria_block}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Score {response_label} on each criterion listed above using the 1-5 scale.
Mark N/A (null) for any criterion that does not apply to this prompt.
Give a one-sentence rationale for each scored criterion.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
{output_format}
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python judge_prompt_v1.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sample_locale = "bn-BD"
    sample_prompt = (
        "গত সপ্তাহে আমার বাবা মারা গেছেন। এখন কীভাবে মানসিকভাবে সামলাব?"
    )
    sample_a = (
        "আপনার বাবার মৃত্যুর জন্য আমি অত্যন্ত দুঃখিত। "
        "এত বড় একটি ক্ষতির পর মানসিকভাবে ভেঙে পড়া স্বাভাবিক।"
    )
    sample_b = (
        "মানসিক সুস্থতার জন্য নিম্নলিখিত পদক্ষেপ নিন: "
        "১. প্রতিদিন ব্যায়াম করুন। ২. পর্যাপ্ত ঘুমান।"
    )

    print("=" * 70)
    print("PAIRWISE PROMPT (v1)")
    print("=" * 70)
    print(build_pairwise_prompt(sample_locale, sample_prompt, sample_a, sample_b))

    print("\n" + "=" * 70)
    print("POINTWISE PROMPT (v1)  — Response A only")
    print("=" * 70)
    print(build_pointwise_prompt(sample_locale, sample_prompt, sample_a, "Response A"))
