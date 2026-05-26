"""
Positional-bias evaluation for LLM-judge results on a pos-bias dataset.

Positional bias: a judge systematically favours a response because of its
position (A or B), not its quality.  Detection: present each pair in both
orderings; a flip in the judge's preference between the original and the
swapped version signals positional bias for that prompt.

Metrics computed
----------------
Overall
  - n_complete_pairs   : prompt_ids with both original & flipped results
  - n_incomplete_pairs : prompt_ids with only one leg (one was skipped)
  - n_biased           : pairs where the judge's underlying choice changed
  - bias_rate          : n_biased / n_complete_pairs
  - consistent_A / B   : judge stuck with position A / B on both legs
  - consistent_correct : judge was correct on BOTH legs
  - position_accuracy  : how often preferred == ground_truth_winner
                         split by position="original" vs position="flipped"

By-dimension breakdowns (scenario_type, primary_criterion, flaw_type, locale)
  - For each category value: total pairs, biased pairs, bias rate
  - Also: accuracy on original leg, accuracy on flipped leg

Usage
-----
  python src/flip_bias_evaluation.py \\
      results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json

From Python:
  from src.flip_bias_evaluation import run
  report = run("results/judge_dev_set_eng_spanish_pos_bias_HuggingFaceTB-SmolLM3-3B_vv0.json")
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

_FLIP_WINNER = {"A": "B", "B": "A"}


def _rate(num: int, denom: int) -> float:
    return num / denom if denom > 0 else 0.0


def _underlying_choice(preferred: str, position: str) -> str:
    """
    Map a judge's stated preference back to the *original* response label.

    In the flipped leg, response_A is what was originally response_B, so
    a judge preference of "A" in the flipped leg means the judge picked the
    original response_B.

    Tie is kept as-is (no meaningful underlying choice).
    """
    if preferred == "Tie":
        return "Tie"
    if position == "flipped":
        return _FLIP_WINNER.get(preferred, preferred)
    return preferred  # original leg: stated == underlying


# ── loading ───────────────────────────────────────────────────────────────────

def load_results(results_path: str) -> tuple[list[dict], dict]:
    """
    Load judge results JSON.

    Parameters
    ----------
    results_path : str
        Path to judge_*_pos_bias_*.json

    Returns
    -------
    tuple[list[dict], dict]
        (records, metadata)
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = {
        "model":         data.get("model", "unknown"),
        "rubric_version": data.get("rubric_version", "unknown"),
        "n_evaluated":   data.get("n_evaluated", 0),
        "n_skipped":     data.get("n_skipped", 0),
        "judge_version": data.get("judge_version", ""),
    }
    return data["results"], metadata


# ── pair building ─────────────────────────────────────────────────────────────

def build_pairs(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Group records by prompt_id into complete (both legs) and incomplete pairs.

    Parameters
    ----------
    records : list[dict]

    Returns
    -------
    tuple[list[dict], list[dict]]
        (complete_pairs, incomplete_singles)

        complete_pairs  — list of dicts:
            {prompt_id, locale, scenario_type, primary_criterion, flaw_type,
             original: record, flipped: record}

        incomplete_singles — list of raw records that have no partner
    """
    by_id: dict[str, dict] = defaultdict(dict)
    for r in records:
        by_id[r["prompt_id"]][r["position"]] = r

    complete = []
    incomplete = []
    for pid, legs in by_id.items():
        if "original" in legs and "flipped" in legs:
            orig = legs["original"]
            complete.append({
                "prompt_id":         pid,
                "locale":            orig.get("locale", "unknown"),
                "scenario_type":     orig.get("scenario_type", "unknown"),
                "primary_criterion": orig.get("primary_criterion", "unknown"),
                "flaw_type":         orig.get("flaw_type", "unknown"),
                "original":          legs["original"],
                "flipped":           legs["flipped"],
            })
        else:
            sole = next(iter(legs.values()))
            incomplete.append(sole)

    return complete, incomplete


# ── core computation ──────────────────────────────────────────────────────────

def compute_bias(pairs: list[dict]) -> dict:
    """
    Compute positional-bias metrics from complete pairs.

    A pair is *biased* when the judge's underlying choice differs between
    the original and flipped legs (i.e. the judge picked a different
    response once the positions changed).

    Parameters
    ----------
    pairs : list[dict]
        Output from build_pairs() complete_pairs list.

    Returns
    -------
    dict
        Overall counts + per-dimension breakdowns.
        Each breakdown value: {key: {"pairs": int, "biased": int,
                                     "acc_original": int, "acc_flipped": int}}
    """
    n_biased = 0
    n_consistent_A = 0    # always picked position A
    n_consistent_B = 0    # always picked position B
    n_correct_both = 0    # correct on both legs

    # accuracy split by position
    acc = {"original": {"correct": 0, "total": 0},
           "flipped":  {"correct": 0, "total": 0}}

    dimensions = ["locale", "scenario_type", "primary_criterion", "flaw_type"]
    breakdown: dict[str, dict] = {
        d: defaultdict(lambda: {"pairs": 0, "biased": 0,
                                "acc_original": 0, "acc_flipped": 0})
        for d in dimensions
    }

    for pair in pairs:
        orig = pair["original"]
        flip = pair["flipped"]

        pref_orig = orig["preferred"]
        pref_flip = flip["preferred"]

        gt_orig = orig.get("ground_truth_winner", "")
        gt_flip = flip.get("ground_truth_winner", "")

        # Underlying choice: what response did the judge actually prefer?
        uc_orig = _underlying_choice(pref_orig, "original")
        uc_flip = _underlying_choice(pref_flip, "flipped")

        is_biased = uc_orig != uc_flip
        if is_biased:
            n_biased += 1

        # Consistency pattern: both legs preferred position A / position B
        if pref_orig == "A" and pref_flip == "A":
            n_consistent_A += 1
        elif pref_orig == "B" and pref_flip == "B":
            n_consistent_B += 1

        # Accuracy per leg
        correct_orig = int(pref_orig == gt_orig and gt_orig != "")
        correct_flip = int(pref_flip == gt_flip and gt_flip != "")

        acc["original"]["total"] += 1
        acc["original"]["correct"] += correct_orig
        acc["flipped"]["total"] += 1
        acc["flipped"]["correct"] += correct_flip

        if correct_orig and correct_flip:
            n_correct_both += 1

        # Per-dimension breakdown
        for d in dimensions:
            key = pair.get(d, "unknown")
            bk = breakdown[d][key]
            bk["pairs"] += 1
            if is_biased:
                bk["biased"] += 1
            bk["acc_original"] += correct_orig
            bk["acc_flipped"] += correct_flip

    return {
        "n_pairs":          len(pairs),
        "n_biased":         n_biased,
        "n_consistent_A":   n_consistent_A,
        "n_consistent_B":   n_consistent_B,
        "n_correct_both":   n_correct_both,
        "acc_by_position":  acc,
        "by_locale":            {k: dict(v) for k, v in breakdown["locale"].items()},
        "by_scenario_type":     {k: dict(v) for k, v in breakdown["scenario_type"].items()},
        "by_primary_criterion": {k: dict(v) for k, v in breakdown["primary_criterion"].items()},
        "by_flaw_type":         {k: dict(v) for k, v in breakdown["flaw_type"].items()},
    }


# ── formatting ────────────────────────────────────────────────────────────────

def _format_bias_table(data: dict, label_header: str,
                       sort_by_biased: bool = False) -> list[str]:
    """
    Render a per-dimension bias breakdown as aligned table lines.

    Columns: label | pairs | biased | bias_rate | acc_orig | acc_flip
    """
    if not data:
        return ["  (no data)"]

    col_w = max(len(label_header), max(len(k) for k in data))
    header = (
        f"  {label_header:<{col_w}}"
        f"  {'pairs':>5}"
        f"  {'biased':>6}"
        f"  {'bias%':>6}"
        f"  {'acc_orig':>8}"
        f"  {'acc_flip':>8}"
    )
    sep = (
        "  " + "-" * col_w
        + "  " + "-" * 5
        + "  " + "-" * 6
        + "  " + "-" * 6
        + "  " + "-" * 8
        + "  " + "-" * 8
    )

    if sort_by_biased:
        rows = sorted(data.items(), key=lambda kv: (-kv[1]["biased"], -kv[1]["pairs"]))
    else:
        rows = sorted(data.items())

    lines = [header, sep]
    for key, v in rows:
        bias_rate = _rate(v["biased"], v["pairs"])
        acc_orig  = _rate(v["acc_original"], v["pairs"])
        acc_flip  = _rate(v["acc_flipped"],  v["pairs"])
        lines.append(
            f"  {key:<{col_w}}"
            f"  {v['pairs']:>5}"
            f"  {v['biased']:>6}"
            f"  {bias_rate:>6.3f}"
            f"  {acc_orig:>8.3f}"
            f"  {acc_flip:>8.3f}"
        )
    return lines


def format_report(metadata: dict, bias: dict,
                  incomplete: list[dict]) -> str:
    """
    Format the full positional-bias report as a string.

    Parameters
    ----------
    metadata : dict
        From load_results()
    bias : dict
        From compute_bias()
    incomplete : list[dict]
        Single-leg records from build_pairs()

    Returns
    -------
    str
        Formatted report text.
    """
    WIDE = "=" * 68
    THIN = "-" * 68
    lines = []

    lines.append("")
    lines.append(WIDE)
    lines.append("Positional Bias Evaluation Report")
    lines.append(WIDE)
    lines.append("")

    # ── Metadata ──────────────────────────────────────────────────────────────
    lines.append(f"Model          : {metadata['model']}")
    lines.append(f"Rubric version : {metadata['rubric_version']}")
    if metadata["judge_version"]:
        lines.append(f"Judge version  : {metadata['judge_version']}")
    lines.append(f"N evaluated    : {metadata['n_evaluated']}")
    lines.append(f"N skipped      : {metadata['n_skipped']}")
    lines.append("")

    # ── Overall ───────────────────────────────────────────────────────────────
    n_pairs   = bias["n_pairs"]
    n_biased  = bias["n_biased"]
    n_cA      = bias["n_consistent_A"]
    n_cB      = bias["n_consistent_B"]
    n_both    = bias["n_correct_both"]
    bias_rate = _rate(n_biased, n_pairs)

    lines.append(THIN)
    lines.append("Overall Bias Summary")
    lines.append(THIN)
    lines.append(f"  Complete pairs         : {n_pairs}")
    lines.append(f"  Incomplete pairs       : {len(incomplete)}"
                 + (f"  {[r['prompt_id'] for r in incomplete]}" if incomplete else ""))
    lines.append("")
    lines.append(f"  Biased pairs           : {n_biased} / {n_pairs}"
                 f"   (bias rate: {bias_rate:.3f})")
    lines.append(f"  Consistent (always A)  : {n_cA}"
                 f"   ({_rate(n_cA, n_pairs):.3f})")
    lines.append(f"  Consistent (always B)  : {n_cB}"
                 f"   ({_rate(n_cB, n_pairs):.3f})")
    lines.append(f"  Correct on BOTH legs   : {n_both}"
                 f"   ({_rate(n_both, n_pairs):.3f})")
    lines.append("")

    # ── Accuracy by position ──────────────────────────────────────────────────
    lines.append(THIN)
    lines.append("Accuracy by Position  (preferred == ground_truth_winner)")
    lines.append(THIN)
    for pos in ("original", "flipped"):
        a = bias["acc_by_position"][pos]
        acc = _rate(a["correct"], a["total"])
        lines.append(f"  {pos:<10} : {a['correct']:>3} / {a['total']:>3}"
                     f"   accuracy: {acc:.3f}")
    lines.append("")

    # ── Per-dimension breakdowns ───────────────────────────────────────────────
    sections = [
        ("Breakdown by Locale",            "locale",            "by_locale",            False),
        ("Breakdown by Scenario Type",     "scenario_type",     "by_scenario_type",     False),
        ("Breakdown by Primary Criterion", "criterion",         "by_primary_criterion", False),
        ("Breakdown by Flaw Type",         "flaw_type",         "by_flaw_type",         True),
    ]

    for title, label_header, key, sort_by in sections:
        lines.append(THIN)
        lines.append(title)
        lines.append(THIN)
        lines.extend(_format_bias_table(bias[key], label_header, sort_by))
        lines.append("")

    # ── Interpretation note ───────────────────────────────────────────────────
    lines.append(THIN)
    lines.append("Notes")
    lines.append(THIN)
    lines.append("  bias%     : fraction of pairs where the judge's underlying")
    lines.append("              choice changed when positions were swapped.")
    lines.append("  acc_orig  : accuracy on the original-order leg.")
    lines.append("  acc_flip  : accuracy on the flipped-order leg.")
    lines.append("  A high bias% with acc_orig >> acc_flip (or vice versa)")
    lines.append("  indicates the judge systematically favours one position.")
    lines.append("")
    lines.append(WIDE)
    lines.append("")

    return "\n".join(lines)


# ── save ──────────────────────────────────────────────────────────────────────

def save_report(report_text: str, results_path: str) -> Path:
    """
    Save report to results/ as posbias_{remainder}.txt.

    Parameters
    ----------
    report_text : str
    results_path : str

    Returns
    -------
    Path
    """
    results_dir = Path(results_path).parent
    stem = Path(results_path).stem
    if stem.startswith("judge_"):
        remainder = stem[len("judge_"):]
        out_name = f"posbias_{remainder}.txt"
    else:
        out_name = "posbias.txt"

    out_path = results_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return out_path


# ── orchestrator ──────────────────────────────────────────────────────────────

def run(results_path: str) -> dict:
    """
    Load → pair → compute → format → print → save.

    Parameters
    ----------
    results_path : str
        Path to judge_*_pos_bias_*.json

    Returns
    -------
    dict
        {metadata, bias, incomplete, report_path}
    """
    records, metadata = load_results(results_path)
    pairs, incomplete = build_pairs(records)
    bias = compute_bias(pairs)
    report_text = format_report(metadata, bias, incomplete)

    print(report_text, end="")

    out_path = save_report(report_text, results_path)
    print(f"Report saved to {out_path}")

    return {
        "metadata":    metadata,
        "bias":        bias,
        "incomplete":  incomplete,
        "report_path": str(out_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/flip_bias_evaluation.py "
              "<path/to/judge_*_pos_bias_*.json> [...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not Path(path).exists():
            print(f"WARNING: File not found: {path}, skipping")
            continue
        run(path)
