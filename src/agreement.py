"""
Compute judge-vs-human agreement metrics: raw accuracy + Cohen's kappa + confusion matrix.

Loads judge_*.json from results/ folder, compares judge.preferred vs ground_truth_winner.
Outputs report to stdout and saves to agr_*.txt in results/ folder.

Usage:
  python src/agreement.py results/judge_test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0.json
  python src/agreement.py results/judge_dev_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0.json

From notebook:
  from src.agreement import run
  metrics = run("results/judge_test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0.json")
  # Prints report to stdout + saves to results/agr_test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0.txt
"""

import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix as sk_confusion_matrix


def load_results(results_path: str) -> tuple[list[str], list[str], dict]:
    """
    Load judge results JSON and extract predictions vs ground truth.

    Parameters
    ----------
    results_path : str
        Path to judge_results_*.json file

    Returns
    -------
    tuple[list[str], list[str], dict]
        (y_pred, y_true, metadata)
        - y_pred: list of 'preferred' values (A, B, Tie)
        - y_true: list of 'ground_truth_winner' values (A, B)
        - metadata: {model, rubric_version, n_evaluated}
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = {
        "model": data.get("model", "unknown"),
        "rubric_version": data.get("rubric_version", "unknown"),
        "n_evaluated": data.get("n_evaluated", 0),
    }

    y_pred = []
    y_true = []

    for result in data["results"]:
        y_pred.append(result["preferred"])
        y_true.append(result["ground_truth_winner"])

    return y_pred, y_true, metadata


def compute_agreement(y_pred: list[str], y_true: list[str]) -> dict:
    """
    Compute raw accuracy and Cohen's kappa.

    Ties are excluded from all computations with a warning.

    Parameters
    ----------
    y_pred : list[str]
        Judge predictions (A, B, Tie)
    y_true : list[str]
        Ground truth (A, B only)

    Returns
    -------
    dict
        {n_total, n_ties, n_used, n_correct, raw_accuracy, cohens_kappa}
    """
    # Count ties
    n_total = len(y_pred)
    tie_indices = [i for i, pred in enumerate(y_pred) if pred == "Tie"]
    n_ties = len(tie_indices)

    if n_ties > 0:
        print(f"WARNING: {n_ties} Tie predictions found and excluded from metrics")

    # Filter out ties
    y_pred_filtered = [p for i, p in enumerate(y_pred) if i not in tie_indices]
    y_true_filtered = [t for i, t in enumerate(y_true) if i not in tie_indices]

    n_used = len(y_pred_filtered)
    n_correct = sum(p == t for p, t in zip(y_pred_filtered, y_true_filtered))
    raw_accuracy = n_correct / n_used if n_used > 0 else 0.0

    # Cohen's kappa (labels must be explicitly provided for consistency)
    kappa = (
        cohen_kappa_score(y_true_filtered, y_pred_filtered, labels=["A", "B"])
        if n_used > 0
        else 0.0
    )

    return {
        "n_total": n_total,
        "n_ties": n_ties,
        "n_used": n_used,
        "n_correct": n_correct,
        "raw_accuracy": raw_accuracy,
        "cohens_kappa": kappa,
    }


def confusion_matrix(y_pred: list[str], y_true: list[str]) -> dict:
    """
    Generate confusion matrix for 2-class case (A vs B).

    Ties are excluded.

    Parameters
    ----------
    y_pred : list[str]
        Judge predictions (A, B, Tie)
    y_true : list[str]
        Ground truth (A, B only)

    Returns
    -------
    dict
        {matrix: [[...]], labels: [...]}
        matrix[i][j] = count of y_true[i] predicted as y_pred[j]
        rows: A, B (human ground truth)
        cols: A, B (judge prediction)
    """
    # Filter out ties
    tie_indices = [i for i, pred in enumerate(y_pred) if pred == "Tie"]
    y_pred_filtered = [p for i, p in enumerate(y_pred) if i not in tie_indices]
    y_true_filtered = [t for i, t in enumerate(y_true) if i not in tie_indices]

    # Compute confusion matrix (rows=y_true, cols=y_pred)
    cm = sk_confusion_matrix(
        y_true_filtered, y_pred_filtered, labels=["A", "B"]
    )

    return {
        "matrix": cm.tolist(),
        "labels": ["A", "B"],
    }


def format_report(metadata: dict, agreement: dict, cm: dict) -> str:
    """
    Format agreement report as a string.

    Parameters
    ----------
    metadata : dict
        {model, rubric_version, n_evaluated}
    agreement : dict
        Output from compute_agreement()
    cm : dict
        Output from confusion_matrix()

    Returns
    -------
    str
        Formatted report text
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("Judge Agreement Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Model          : {metadata['model']}")
    lines.append(f"Rubric version : {metadata['rubric_version']}")
    n_eval_line = f"N evaluated    : {metadata['n_evaluated']}"
    if agreement["n_ties"] > 0:
        n_eval_line += f"  ({agreement['n_used']} used, {agreement['n_ties']} Tie excluded)"
    lines.append(n_eval_line)

    lines.append("")
    lines.append(f"Raw accuracy   : {agreement['raw_accuracy']:.3f}  ({agreement['n_correct']}/{agreement['n_used']})")
    lines.append(f"Cohen's kappa  : {agreement['cohens_kappa']:.3f}")

    lines.append("")
    lines.append("Confusion Matrix (rows=human, cols=judge)")
    labels = cm["labels"]
    matrix = cm["matrix"]

    # Header
    lines.append("        " + "  ".join(f"{l:>3}" for l in labels))

    # Rows
    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{matrix[i][j]:>3}" for j in range(len(labels)))
        lines.append(f"  {label}    {row_vals}")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def print_report(metadata: dict, agreement: dict, cm: dict) -> None:
    """
    Print agreement report to stdout.

    Parameters
    ----------
    metadata : dict
        {model, rubric_version, n_evaluated}
    agreement : dict
        Output from compute_agreement()
    cm : dict
        Output from confusion_matrix()
    """
    report_text = format_report(metadata, agreement, cm)
    print(report_text, end="")


def save_report(report_text: str, results_path: str) -> Path:
    """
    Save report to a text file in results folder.

    Output filename: agr_{dataset_stem}_{model_slug}_v{rubric_version}.txt

    Parameters
    ----------
    report_text : str
        Formatted report text from format_report()
    results_path : str
        Original path to judge_results_*.json (used to derive output filename)

    Returns
    -------
    Path
        Path to saved report file
    """
    results_dir = Path(results_path).parent
    results_file = Path(results_path).stem  # e.g. "judge_test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0"

    # Extract remainder after "judge_" prefix, then mirror into agr_ filename
    if results_file.startswith("judge_"):
        remainder = results_file[len("judge_"):]  # e.g. "test_set_synthetic_prompts_claude-haiku-4-5-20251001_v1.0"
        output_filename = f"agr_{remainder}.txt"
    else:
        output_filename = "agr.txt"

    output_path = results_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path


def run(results_path: str) -> dict:
    """
    Orchestrate: load results → compute metrics → print + save report.

    Parameters
    ----------
    results_path : str
        Path to judge_results_*.json file

    Returns
    -------
    dict
        Combined metrics dict for programmatic use
    """
    y_pred, y_true, metadata = load_results(results_path)
    agreement = compute_agreement(y_pred, y_true)
    cm = confusion_matrix(y_pred, y_true)

    report_text = format_report(metadata, agreement, cm)

    # Print to stdout
    print(report_text, end="")

    # Save to file
    report_path = save_report(report_text, results_path)
    print(f"Report saved to {report_path}")

    # Return all metrics combined
    return {
        "metadata": metadata,
        "agreement": agreement,
        "confusion_matrix": cm,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python src/agreement.py <path/to/judge_results_*.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    if not Path(results_path).exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)

    run(results_path)
