"""
Prepare positional-bias evaluation dataset from a judge CSV.

For every row in the source CSV, emit two rows:
  - original  : responses in original A/B order   (position="original")
  - flipped   : response_A and response_B swapped  (position="flipped")

The `ground_truth_winner` is adjusted in the flipped row so it always names
the *correct* response regardless of which position it sits in:
  - original A → flipped B  (ground_truth_winner: A → B)
  - original B → flipped A  (ground_truth_winner: B → A)

The `flipped` column is rewritten to reflect the current row's swap state
(False for original rows, True for flipped rows), replacing whatever value
the source file carried.

Output: <source_stem>_pos_bias.csv written to the same directory as the input.

Usage:
  python src/prepare_data_pos_bias.py data/dev_set_eng_spanish.csv

From Python:
  from src.prepare_data_pos_bias import run
  out_path = run("data/dev_set_eng_spanish.csv")
"""

import csv
import sys
from pathlib import Path


# ── constants ────────────────────────────────────────────────────────────────

_FLIP_WINNER = {"A": "B", "B": "A"}

_COLUMNS = [
    "prompt_id",
    "locale",
    "scenario_type",
    "primary_criterion",
    "flaw_type",
    "prompt_text",
    "prompt_text_en",
    "response_A",
    "response_A_en",
    "response_B",
    "response_B_en",
    "flipped",
    "ground_truth_winner",
    "position",          # new: "original" | "flipped"
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _original_row(row: dict) -> dict:
    """Return the row as-is with position='original' and flipped=False."""
    out = dict(row)
    out["flipped"] = "False"
    out["position"] = "original"
    return out


def _flipped_row(row: dict) -> dict:
    """Return a new row with response_A ↔ response_B swapped and winner adjusted."""
    out = dict(row)
    # Swap responses
    out["response_A"]    = row["response_B"]
    out["response_A_en"] = row["response_B_en"]
    out["response_B"]    = row["response_A"]
    out["response_B_en"] = row["response_A_en"]
    # Adjust ground truth (A→B, B→A); leave blank/unknown values unchanged
    winner = row.get("ground_truth_winner", "")
    out["ground_truth_winner"] = _FLIP_WINNER.get(winner, winner)
    # Mark as flipped
    out["flipped"] = "True"
    out["position"] = "flipped"
    return out


# ── core functions ────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> list[dict]:
    """
    Load rows from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to source CSV (e.g. data/dev_set_eng_spanish.csv)

    Returns
    -------
    list[dict]
        One dict per row, keyed by column name.
    """
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_pos_bias_rows(rows: list[dict]) -> list[dict]:
    """
    Double every input row: original order first, then flipped.

    For each source row the output contains:
      1. original row  (position="original", flipped=False)
      2. flipped row   (position="flipped",  flipped=True,
                        response_A/B swapped, ground_truth_winner inverted)

    Parameters
    ----------
    rows : list[dict]
        Source rows from load_csv()

    Returns
    -------
    list[dict]
        2 × len(rows) output rows in interleaved original/flipped order.
    """
    out = []
    for row in rows:
        out.append(_original_row(row))
        out.append(_flipped_row(row))
    return out


def save_csv(rows: list[dict], out_path: Path) -> None:
    """
    Write rows to a CSV file using the canonical column order.

    Extra keys present in rows but absent from _COLUMNS are silently dropped.

    Parameters
    ----------
    rows : list[dict]
        Rows produced by build_pos_bias_rows()
    out_path : Path
        Destination file path (created or overwritten).
    """
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(csv_path: str) -> Path:
    """
    Orchestrate: load → build doubled rows → save.

    Parameters
    ----------
    csv_path : str
        Path to source CSV file.

    Returns
    -------
    Path
        Path to the written output file.
    """
    src = Path(csv_path)
    out_path = src.parent / f"{src.stem}_pos_bias.csv"

    rows = load_csv(csv_path)
    doubled = build_pos_bias_rows(rows)
    save_csv(doubled, out_path)

    n_src = len(rows)
    n_out = len(doubled)
    print(f"Source rows    : {n_src}")
    print(f"Output rows    : {n_out}  ({n_src} original + {n_src} flipped)")
    print(f"Saved to       : {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/prepare_data_pos_bias.py <path/to/dataset.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    run(csv_path)
