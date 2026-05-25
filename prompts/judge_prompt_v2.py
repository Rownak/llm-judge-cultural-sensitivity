"""
Judge prompt template for rubric v2 (judge_rubric_v2.json).

v2 field shape differences from v1:
  - dimension["flaws"]            → list[dict]  {label, severity, description}
  - dimension["applicable_when"]  → str
  - dimension["non_applicable_when"] → str
  - dimension["score_anchors"]    → dict {"5": str, "3": str, "1": str}
  - No top-level "flaw_taxonomy"  (flaw metadata inlined into each criterion)
  - No dimension-level "severity" (severity is per-flaw)
  - top-level "judge_instructions" → list[str]
  - pairwise output: flaws_in_A/B → list[dict] {criterion, flaw, severity}
  - pairwise output: "rubric_version" field required
  - pairwise output: "both_defective" boolean field required
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Load rubric
# ---------------------------------------------------------------------------

RUBRIC_PATH = Path(__file__).parent.parent / "rubrics" / "judge_rubric_v2.json"

with RUBRIC_PATH.open(encoding="utf-8") as f:
    RUBRIC = json.load(f)

CRITERIA: dict[str, dict] = {d["id"]: d for d in RUBRIC["dimensions"]}
RUBRIC_VERSION = RUBRIC["version"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_applicable(criterion_id: str, has_emotional_content: bool, is_cultural_topic: bool) -> bool:
    """
    Lightweight applicability check based on rubric applicable_when/non_applicable_when text.
    Callers can also pass active_criteria directly to bypass this.
    """
    dim = CRITERIA[criterion_id]
    naw = dim.get("non_applicable_when", "").lower()

    if criterion_id in ("EA-1", "EA-3") and not has_emotional_content:
        return False
    if criterion_id == "EA-2" and "purely neutral" in naw and not has_emotional_content:
        return False
    if criterion_id == "CS-3" and not is_cultural_topic:
        return False
    return True


def _format_flaws_for_criterion(criterion_id: str) -> str:
    """Bullet list of inline flaw objects (v2 schema: each flaw has label, severity, description)."""
    dim = CRITERIA[criterion_id]
    lines = []
    for flaw in dim["flaws"]:
        lines.append(
            f"    - [{flaw['severity'].upper()}] `{flaw['label']}`: {flaw['description']}"
        )
    return "\n".join(lines)


def _format_score_anchors(criterion_id: str) -> str:
    anchors = CRITERIA[criterion_id].get("score_anchors", {})
    if not anchors:
        return "    (No criterion-specific anchors — use global pointwise scale.)"
    lines = []
    for score_key in ("5", "3", "1"):
        if score_key in anchors:
            lines.append(f"    {score_key} → {anchors[score_key]}")
    return "\n".join(lines)


def _format_locale_note(criterion_id: str, locale: str) -> str:
    return CRITERIA[criterion_id]["locale_notes"].get(locale, "No specific note for this locale.")


def _judge_instructions_block() -> str:
    instructions = RUBRIC.get("judge_instructions", [])
    return "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(instructions))


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
# Core builder — pairwise
# ---------------------------------------------------------------------------

def build_pairwise_prompt(
    locale: str,
    prompt_text: str,
    response_a: str,
    response_b: str,
    active_criteria: list[str] | None = None,
    has_emotional_content: bool | None = None,
    is_cultural_topic: bool | None = None,
) -> str:
    """
    Build a pairwise judge prompt for rubric v2.

    Parameters
    ----------
    locale               : one of "en-US", "bn-BD", "ja-JP"
    prompt_text          : the original user-facing prompt being evaluated
    response_a           : first AI response (labelled A)
    response_b           : second AI response (labelled B)
    active_criteria      : explicit subset of criterion IDs; if None, applicability
                           is auto-detected via has_emotional_content / is_cultural_topic
    has_emotional_content: hint for EA-1/EA-2/EA-3 applicability (auto if None)
    is_cultural_topic    : hint for CS-3 applicability (auto if None)
    """
    all_ids = [d["id"] for d in RUBRIC["dimensions"]]

    if active_criteria is None:
        # auto-detect applicability; default to True when hints not provided
        emo = has_emotional_content if has_emotional_content is not None else True
        cult = is_cultural_topic if is_cultural_topic is not None else True
        active_criteria = [
            cid for cid in all_ids if _is_applicable(cid, emo, cult)
        ]

    skipped = [cid for cid in all_ids if cid not in active_criteria]

    # --- criteria section -------------------------------------------------
    criteria_block_lines = []
    for cid in active_criteria:
        dim = CRITERIA[cid]
        criteria_block_lines.append(
            f"### {cid} · {dim['name']}\n"
            f"Applicable when  : {dim['applicable_when']}\n"
            f"Definition       : {dim['definition']}\n"
            f"Locale note ({locale}): {_format_locale_note(cid, locale)}\n"
            f"Flaws to watch for:\n{_format_flaws_for_criterion(cid)}"
        )
    criteria_block = "\n\n".join(criteria_block_lines)

    skipped_note = ""
    if skipped:
        skipped_note = (
            "\n⚠ Criteria skipped as NOT APPLICABLE for this prompt: "
            + ", ".join(skipped)
            + "\n"
        )

    # --- output format section --------------------------------------------
    output_schema = RUBRIC["pairwise_output_schema"]
    flaw_obj_example = '{"criterion": "EA-1", "flaw": "premature_pivot_to_advice", "severity": "moderate"}'
    output_format = (
        "Return ONLY valid JSON — no prose before or after — matching this schema:\n\n"
        "```json\n"
        "{\n"
        f'  "rubric_version": "{RUBRIC_VERSION}",\n'
        f'  "preferred": {json.dumps(output_schema["preferred"]["enum"])},\n'
        f'  "confidence": {json.dumps(output_schema["confidence"]["enum"])},\n'
        '  "rationale": "<one to three sentences citing specific criterion IDs>",\n'
        '  "flaws_in_A": [ /* flaw objects or empty array */ ],\n'
        '  "flaws_in_B": [ /* flaw objects or empty array */ ],\n'
        '  "both_defective": false\n'
        "}\n"
        "```\n\n"
        f"Each flaw object must have this shape: {flaw_obj_example}\n"
        f"severity must be one of: {output_schema['flaws_in_A']['items']['properties']['severity']['enum']}\n\n"
        f"both_defective: set to true when BOTH responses contain at least one CRITICAL flaw.\n"
        f"Tie policy: {RUBRIC['tie_policy']}\n\n"
        "Reference example output:\n"
        + json.dumps(output_schema["example_output"], ensure_ascii=False, indent=2)
    )

    # --- assemble ---------------------------------------------------------
    prompt = f"""You are a calibrated cultural-sensitivity and emotional-appropriateness judge.
Rubric version: {RUBRIC_VERSION}

═══════════════════════════════════════════════════════
JUDGE INSTRUCTIONS  (read before evaluating)
═══════════════════════════════════════════════════════
{_judge_instructions_block()}

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
{skipped_note}
═══════════════════════════════════════════════════════
APPLICABLE EVALUATION CRITERIA
═══════════════════════════════════════════════════════
{criteria_block}

═══════════════════════════════════════════════════════
SEVERITY WEIGHTING GUIDE
═══════════════════════════════════════════════════════
{_severity_table()}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
For each APPLICABLE criterion above:
  1. Check both responses for the listed flaws.
  2. Note the flaw label and its severity.
  3. Weigh flaws using the severity guide.

Then choose the better response (A, B, or Tie) and explain why,
citing at least one criterion ID in your rationale.
Set both_defective: true if both responses have a CRITICAL flaw.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
{output_format}
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Core builder — pointwise
# ---------------------------------------------------------------------------

def build_pointwise_prompt(
    locale: str,
    prompt_text: str,
    response: str,
    response_label: str = "Response",
    active_criteria: list[str] | None = None,
    has_emotional_content: bool | None = None,
    is_cultural_topic: bool | None = None,
) -> str:
    """
    Build a pointwise 1-5 scoring prompt for rubric v2.

    Parameters
    ----------
    locale               : one of "en-US", "bn-BD", "ja-JP"
    prompt_text          : the original user-facing prompt being evaluated
    response             : the single AI response to score
    response_label       : display label (e.g. "Response A")
    active_criteria      : explicit subset of criterion IDs; auto-detected if None
    has_emotional_content: hint for EA criteria applicability
    is_cultural_topic    : hint for CS-3 applicability
    """
    all_ids = [d["id"] for d in RUBRIC["dimensions"]]

    if active_criteria is None:
        emo = has_emotional_content if has_emotional_content is not None else True
        cult = is_cultural_topic if is_cultural_topic is not None else True
        active_criteria = [
            cid for cid in all_ids if _is_applicable(cid, emo, cult)
        ]

    skipped = [cid for cid in all_ids if cid not in active_criteria]

    # --- criteria section -------------------------------------------------
    criteria_lines = []
    for cid in active_criteria:
        dim = CRITERIA[cid]
        criteria_lines.append(
            f"  {cid} · {dim['name']}\n"
            f"    Applicable when    : {dim['applicable_when']}\n"
            f"    Non-applicable when: {dim['non_applicable_when']}\n"
            f"    Definition         : {dim['definition']}\n"
            f"    Locale note        : {_format_locale_note(cid, locale)}\n"
            f"    Flaws:\n{_format_flaws_for_criterion(cid)}\n"
            f"    Score anchors:\n{_format_score_anchors(cid)}"
        )
    criteria_block = "\n\n".join(criteria_lines)

    skipped_note = ""
    if skipped:
        skipped_note = (
            "\n⚠ Criteria NOT APPLICABLE for this prompt (mark null): "
            + ", ".join(skipped)
            + "\n"
        )

    # --- score output section ---------------------------------------------
    score_fields = "\n".join(
        f'  "{cid}": {{"score": <1–5 or null>, "rationale": "<one sentence>", "flaws": []}}'
        for cid in active_criteria
    )
    output_format = (
        "Return ONLY valid JSON:\n\n"
        "```json\n"
        "{\n"
        f'  "rubric_version": "{RUBRIC_VERSION}",\n'
        f"{score_fields}\n"
        "}\n"
        "```\n\n"
        "For each scored criterion, 'flaws' is an array of flaw objects:\n"
        '  {"criterion": "<id>", "flaw": "<label>", "severity": "<critical|moderate|minor>"}\n\n'
        f"Use null for score when the criterion is not applicable.\n\n"
        f"Score meanings (global scale):\n{_pointwise_scale()}\n\n"
        "Criterion-specific anchors are listed above each criterion — use them to calibrate."
    )

    prompt = f"""You are a calibrated cultural-sensitivity and emotional-appropriateness judge.
Rubric version: {RUBRIC_VERSION}

═══════════════════════════════════════════════════════
JUDGE INSTRUCTIONS  (read before evaluating)
═══════════════════════════════════════════════════════
{_judge_instructions_block()}

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
{skipped_note}
═══════════════════════════════════════════════════════
APPLICABLE EVALUATION CRITERIA  (with criterion-specific score anchors)
═══════════════════════════════════════════════════════
{criteria_block}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Score {response_label} on each APPLICABLE criterion using the 1–5 scale.
  • Use the criterion-specific score anchors above to calibrate.
  • Mark null for any criterion listed as not applicable.
  • List every flaw found in the 'flaws' array for that criterion.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
{output_format}
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python judge_prompt_v2.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sample_locale = "en-US"
    sample_prompt = (
        "My boss pointed out the same mistake again. Every time it's the same feedback and I'm getting frustrated."
    )
    sample_a = (
        "That sounds really frustrating. When you receive repeated feedback on the same issue, it's natural to feel discouraged. "
        "Your feelings are valid, and many people experience this."
    )
    sample_b = (
        "Your boss is giving you this feedback because they see potential in you and want you to grow. "
        "Try to view it positively and work harder to eliminate these mistakes!"
    )

    print("=" * 70)
    print("PAIRWISE PROMPT (v2)")
    print("=" * 70)
    print(
        build_pairwise_prompt(
            sample_locale,
            sample_prompt,
            sample_a,
            sample_b,
            has_emotional_content=True,
            is_cultural_topic=False,
        )
    )

    print("\n" + "=" * 70)
    print("POINTWISE PROMPT (v2)  — Response B only")
    print("=" * 70)
    print(
        build_pointwise_prompt(
            sample_locale,
            sample_prompt,
            sample_b,
            "Response B",
            has_emotional_content=True,
            is_cultural_topic=False,
        )
    )
