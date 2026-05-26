"""
Simple judge prompt (v0.1) - No rubric, just cultural diversity & emotional appropriateness.

Identical to v0 except the pairwise prompt adds an explicit anti-positional-bias
instruction: the judge is told to evaluate response quality independently of
whether a response is labelled A or B, or which one appears first.

Output format mirrors v1 pairwise schema but with freeform flaw descriptions
(no criterion IDs/labels).
"""

import json

RUBRIC_VERSION = "0.1"

def build_pairwise_prompt(
    locale: str,
    prompt_text: str,
    response_a: str,
    response_b: str,
) -> str:
    """
    Build a simple pairwise judge prompt without rubric.

    Differs from v0: adds a IMPORTANT INSTRUCTION section that explicitly
    instructs the model not to prefer a response based on its position
    (A vs B, or first vs second presented).

    Parameters
    ----------
    locale      : the locale/region being evaluated (e.g., "en-US", "bn-BD", "ja-JP")
    prompt_text : the original user-facing prompt being evaluated
    response_a  : first AI response (labelled A)
    response_b  : second AI response (labelled B)
    """

    prompt = f"""You are a judge evaluating two AI responses for cultural sensitivity and emotional appropriateness.

═══════════════════════════════════════════════════════
IMPORTANT INSTRUCTION — AVOID POSITIONAL BIAS
═══════════════════════════════════════════════════════
Do NOT favour a response simply because it is labelled A or B,
or because it appears first or second.
The labels A and B are arbitrary — evaluate each response solely
on its content, quality, cultural sensitivity, and emotional
appropriateness. Your verdict must be the same regardless of
which position a response occupies.

═══════════════════════════════════════════════════════
LOCALE / REGION UNDER EVALUATION
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
OUTPUT FORMAT
═══════════════════════════════════════════════════════
Return ONLY valid JSON — no prose before or after — matching this schema:

```json
{{
  "preferred": "A" | "B" | "Tie",
  "confidence": "high" | "medium" | "low",
  "rationale": "<2-3 sentences explaining the choice>",
  "flaws_in_A": ["<flaw description>", ...],
  "flaws_in_B": ["<flaw description>", ...]
}}
```
"""
    return prompt.strip()


def build_pointwise_prompt(
    locale: str,
    prompt_text: str,
    response: str,
    response_label: str = "Response",
) -> str:
    """
    Build a simple pointwise 1-5 scoring prompt without rubric.

    Unchanged from v0 (positional bias is not applicable to pointwise scoring).

    Parameters
    ----------
    locale         : the locale/region being evaluated
    prompt_text    : the original user-facing prompt being evaluated
    response       : the single AI response to score
    response_label : display label (e.g. "Response A")
    """

    prompt = f"""You are a judge evaluating an AI response for cultural sensitivity and emotional appropriateness.

═══════════════════════════════════════════════════════
LOCALE / REGION UNDER EVALUATION
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
1. **Cultural Sensitivity**: Does the response respect and acknowledge diverse cultural backgrounds, values, and norms relevant to the locale?

2. **Emotional Appropriateness**: Does the response demonstrate genuine empathy, emotional awareness, and sensitivity to the user's emotional state?

3. **Inclusivity**: Is the language and tone inclusive, avoiding stereotypes, assumptions, or culturally insensitive language?

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Score {response_label} on cultural sensitivity and emotional appropriateness using the 1-5 scale below.
Identify any flaws related to cultural insensitivity or emotional inappropriateness.

═══════════════════════════════════════════════════════
SCORING SCALE
═══════════════════════════════════════════════════════
1 – Poor: Culturally insensitive, dismissive, or emotionally inappropriate
2 – Below Average: Some cultural insensitivity or emotional gaps
3 – Adequate: Acceptable cultural sensitivity and emotional appropriateness
4 – Good: Demonstrates cultural awareness and emotional empathy
5 – Excellent: Highly culturally sensitive, emotionally aware, and inclusive

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
Return ONLY valid JSON:

```json
{{
  "score": <1-5>,
  "rationale": "<1-2 sentences explaining the score>",
  "flaws": ["<flaw description>", ...]
}}
```

Use empty list [] if no flaws were found.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Quick smoke-test  (python prompts/judge_prompt_v0_1.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    sample_locale = "en-US"
    sample_prompt = (
        "My father passed away last week. How do I cope with this mentally and emotionally?"
    )
    sample_a = (
        "I'm deeply sorry for your loss. It's completely natural to feel overwhelmed after such a profound loss. "
        "Please be gentle with yourself during this time."
    )
    sample_b = (
        "Here are some steps for mental wellness: 1. Exercise daily. 2. Get enough sleep. 3. Stay busy with activities."
    )

    print("=" * 70)
    print("PAIRWISE PROMPT (v0.1)")
    print("=" * 70)
    print(build_pairwise_prompt(sample_locale, sample_prompt, sample_a, sample_b))

    print("\n" + "=" * 70)
    print("POINTWISE PROMPT (v0.1)")
    print("=" * 70)
    print(build_pointwise_prompt(sample_locale, sample_prompt, sample_a, "Response A"))
