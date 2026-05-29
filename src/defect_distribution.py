"""
Compute defect distribution: where do judge mistakes cluster?

A "defect" is a judge mistake: preferred != ground_truth_winner.
Ties count as defects (ground truth is always A or B).
Only records that contain ground_truth_winner are analysed.

Breakdowns: locale, scenario_type, primary_criterion, flaw_type, confidence.

Usage:
  python src/defect_distribution.py results/judge_*.json
  python src/defect_distribution.py results/judge_dev_set_eng_spanish_HuggingFaceTB-SmolLM3-3B_vv0.json

From notebook:
  from src.defect_distribution import run
  report = run("results/judge_dev_set_eng_spanish_HuggingFaceTB-SmolLM3-3B_vv0.json")
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_path: str) -> tuple[list[dict], dict]:
    """
    Load judge results JSON and extract records and metadata.

    Parameters
    ----------
    results_path : str
        Path to judge_*.json file

    Returns
    -------
    tuple[list[dict], dict]
        (records, metadata)
        - records: list of individual evaluation dicts
        - metadata: {model, rubric_version, n_evaluated, prompt_version}
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = {
        "model": data.get("model", "unknown"),
        "rubric_version": data.get("rubric_version", "unknown"),
        "n_evaluated": data.get("n_evaluated", 0),
        "prompt_version": data.get("prompt_version", data.get("judge_version", "")),
    }

    return data["results"], metadata


def _defect_rate(defects: int, total: int) -> float:
    """Return defects / total, or 0.0 if total is zero."""
    return defects / total if total > 0 else 0.0


def compute_defects(records: list[dict]) -> dict:
    """
    Compute defect counts broken down by key dimensions.

    A defect is any record where preferred != ground_truth_winner.
    Tie predictions count as defects (ground truth is always A or B).
    Records without ground_truth_winner are excluded from all counts.

    Parameters
    ----------
    records : list[dict]
        List of evaluation result dicts from the JSON results array

    Returns
    -------
    dict
        {
            n_total, n_defects, n_ties,
            by_locale, by_scenario_type, by_primary_criterion,
            by_confidence, by_flaw_type
        }
        Each by_* value is a dict of {key: {"total": int, "defects": int}}
    """
    records_with_gt = [r for r in records if r.get("ground_truth_winner")]
    defects_list = [
        r for r in records_with_gt if r["preferred"] != r["ground_truth_winner"]
    ]
    defect_ids = {id(r) for r in defects_list}
    n_ties = sum(1 for r in records_with_gt if r["preferred"] == "Tie")

    dimensions = {
        "by_locale": "locale",
        "by_scenario_type": "scenario_type",
        "by_primary_criterion": "primary_criterion",
        "by_confidence": "confidence",
        "by_flaw_type": "flaw_type",
    }

    breakdown = {
        dim_key: defaultdict(lambda: {"total": 0, "defects": 0})
        for dim_key in dimensions
    }

    for r in records_with_gt:
        is_defect = id(r) in defect_ids
        for dim_key, field in dimensions.items():
            key = r.get(field, "unknown")
            breakdown[dim_key][key]["total"] += 1
            if is_defect:
                breakdown[dim_key][key]["defects"] += 1

    return {
        "n_total": len(records),
        "n_defects": len(defects_list),
        "n_ties": n_ties,
        "by_locale": dict(breakdown["by_locale"]),
        "by_scenario_type": dict(breakdown["by_scenario_type"]),
        "by_primary_criterion": dict(breakdown["by_primary_criterion"]),
        "by_confidence": dict(breakdown["by_confidence"]),
        "by_flaw_type": dict(breakdown["by_flaw_type"]),
    }


def _format_table(
    data: dict,
    label_header: str,
    sort_by_defects: bool = False,
) -> list[str]:
    """
    Format a dimension breakdown dict as aligned table lines.

    Parameters
    ----------
    data : dict
        {key: {"total": int, "defects": int}}
    label_header : str
        Column header for the key column (e.g. "locale")
    sort_by_defects : bool
        If True, sort rows by defects DESC then total DESC.
        If False, sort alphabetically by key.

    Returns
    -------
    list[str]
        Lines of the table (header row + data rows), no trailing newline.
    """
    if not data:
        return ["  (no data)"]

    col_width = max(len(label_header), max(len(k) for k in data))

    header = (
        f"  {label_header:<{col_width}}  {'total':>6}  {'defects':>7}  {'rate':>6}"
    )
    separator = "  " + "-" * col_width + "  " + "-" * 6 + "  " + "-" * 7 + "  " + "-" * 6

    if sort_by_defects:
        rows = sorted(data.items(), key=lambda kv: (-kv[1]["defects"], -kv[1]["total"]))
    else:
        rows = sorted(data.items())

    lines = [header, separator]
    for key, counts in rows:
        rate = _defect_rate(counts["defects"], counts["total"])
        lines.append(
            f"  {key:<{col_width}}  {counts['total']:>6}  {counts['defects']:>7}  {rate:>6.3f}"
        )
    return lines


def format_report(metadata: dict, defects: dict) -> str:
    """
    Format the defect distribution report as a string.

    Parameters
    ----------
    metadata : dict
        {model, rubric_version, n_evaluated, prompt_version}
    defects : dict
        Output from compute_defects()

    Returns
    -------
    str
        Full formatted report text
    """
    lines = []
    WIDE = "=" * 60
    THIN = "-" * 60

    lines.append("")
    lines.append(WIDE)
    lines.append("Defect Distribution Report")
    lines.append(WIDE)
    lines.append("")

    # Header metadata
    lines.append(f"Model          : {metadata['model']}")
    lines.append(f"Rubric version : {metadata['rubric_version']}")
    if metadata["prompt_version"]:
        lines.append(f"Prompt version : {metadata['prompt_version']}")
    lines.append(f"N total        : {defects['n_total']}")

    rate = _defect_rate(defects["n_defects"], defects["n_total"])
    tie_note = f"  [{defects['n_ties']} Tie]" if defects["n_ties"] > 0 else ""
    lines.append(
        f"N defects      : {defects['n_defects']}   (rate: {rate:.3f}){tie_note}"
    )
    lines.append("")

    # Breakdown sections
    sections = [
        ("Breakdown by Locale", "locale", "by_locale", False),
        ("Breakdown by Scenario Type", "scenario_type", "by_scenario_type", False),
        ("Breakdown by Primary Criterion", "criterion", "by_primary_criterion", False),
        ("Breakdown by Flaw Type", "flaw_type", "by_flaw_type", True),
        ("Breakdown by Confidence", "confidence", "by_confidence", False),
    ]

    for title, label_header, key, sort_by_defects in sections:
        lines.append(THIN)
        lines.append(title)
        lines.append(THIN)
        lines.extend(_format_table(defects[key], label_header, sort_by_defects))
        lines.append("")

    lines.append(WIDE)
    lines.append("")

    return "\n".join(lines)


def print_report(metadata: dict, defects: dict) -> None:
    """
    Print defect distribution report to stdout.

    Parameters
    ----------
    metadata : dict
        {model, rubric_version, n_evaluated, prompt_version}
    defects : dict
        Output from compute_defects()
    """
    report_text = format_report(metadata, defects)
    print(report_text, end="")


def save_report(report_text: str, results_path: str, output_dir: Path | None = None) -> Path:
    """
    Save report to a text file in the results/defects folder.

    Output filename: defects_{dataset_stem}.txt
    (mirrors agreement.py which uses agr_{dataset_stem}.txt)

    Parameters
    ----------
    report_text : str
        Formatted report text from format_report()
    results_path : str
        Path to the source judge_*.json file
    output_dir : Path | None
        Directory to write the report. Defaults to results/defects/ relative to project root.

    Returns
    -------
    Path
        Path to the saved report file
    """
    stem = Path(results_path).stem  # e.g. "judge_dev_set_eng_spanish_..."

    if stem.startswith("judge_"):
        remainder = stem[len("judge_"):]
        output_filename = f"defects_{remainder}.txt"
    else:
        output_filename = "defects.txt"

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "defects"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path


def run(results_path: str, output_dir: Path | None = None) -> dict:
    """
    Orchestrate: load results → compute defects → print + save report.

    Parameters
    ----------
    results_path : str
        Path to judge_*.json file

    Returns
    -------
    dict
        {metadata, defects, report_path}
    """
    records, metadata = load_results(results_path)
    defects = compute_defects(records)
    report_text = format_report(metadata, defects)

    print(report_text, end="")

    report_path = save_report(report_text, results_path, output_dir=output_dir)
    print(f"Report saved to {report_path}")

    return {
        "metadata": metadata,
        "defects": defects,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute defect distribution from judge results")
    parser.add_argument("results_paths", nargs="+", help="Path(s) to judge_*.json file(s)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write defects_*.txt (default: results/defects/ relative to project root)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    for path in args.results_paths:
        if not Path(path).exists():
            print(f"WARNING: File not found: {path}, skipping")
            continue
        run(path, output_dir=output_dir)
