"""
Prepare dataset for pairwise LLM judge evaluation.

- Load synthetic_prompts.csv
- Create 80/20 dev/test split (stratified by locale)
- Randomize position A/B for each pair
- Generate ground-truth labels for test set
- Output two CSV files: dev_set.csv and test_set.csv
"""

import csv
import random
from pathlib import Path
from collections import defaultdict


def load_prompts(csv_path):
    """Load synthetic prompts from CSV."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f, skipinitialspace=True, quoting=csv.QUOTE_ALL)
        for row in reader:
            # Clean up keys: remove quotes if present
            row = {k.strip('"'): v for k, v in row.items()}
            rows.append(row)
    return rows


def assign_position(row, seed):
    """
    Randomly assign good/flawed responses to positions A/B.

    Returns dict with:
      - response_A, response_B, response_A_en, response_B_en
      - ground_truth_winner: "A" or "B" (which is the good response)
      - flipped: bool (True if good is in position B)
    """
    rng = random.Random(seed)
    is_flipped = rng.random() < 0.5

    if is_flipped:
        return {
            "response_A": row["response_flawed"],
            "response_A_en": row["response_flawed_en"],
            "response_B": row["response_good"],
            "response_B_en": row["response_good_en"],
            "ground_truth_winner": "B",
            "flipped": True,
        }
    else:
        return {
            "response_A": row["response_good"],
            "response_A_en": row["response_good_en"],
            "response_B": row["response_flawed"],
            "response_B_en": row["response_flawed_en"],
            "ground_truth_winner": "A",
            "flipped": False,
        }


def stratified_split(rows, test_ratio=0.2, seed=42):
    """
    Split rows into dev/test maintaining locale distribution.
    """
    random.seed(seed)

    # Group by locale
    by_locale = defaultdict(list)
    for row in rows:
        by_locale[row["locale"]].append(row)

    dev_set = []
    test_set = []

    # For each locale, split 80/20
    for locale, locale_rows in by_locale.items():
        random.shuffle(locale_rows)
        split_idx = int(len(locale_rows) * (1 - test_ratio))
        dev_set.extend(locale_rows[:split_idx])
        test_set.extend(locale_rows[split_idx:])

    # Shuffle final sets
    random.shuffle(dev_set)
    random.shuffle(test_set)

    return dev_set, test_set


def prepare_dataset(input_csv, output_dir, test_ratio=0.2, seed=42):
    """
    Prepare dev and test sets with randomized positions and ground truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading {input_csv}...")
    rows = load_prompts(input_csv)
    print(f"  Loaded {len(rows)} prompts")

    # Split
    print(f"Splitting {test_ratio*100:.0f}% test, {(1-test_ratio)*100:.0f}% dev...")
    dev_set, test_set = stratified_split(rows, test_ratio=test_ratio, seed=seed)
    print(f"  Dev: {len(dev_set)} items")
    print(f"  Test: {len(test_set)} items")

    # Check locale distribution
    for name, dataset in [("Dev", dev_set), ("Test", test_set)]:
        locale_counts = defaultdict(int)
        for row in dataset:
            locale_counts[row["locale"]] += 1
        print(f"  {name} locale distribution: {dict(locale_counts)}")

    # Prepare position assignment and write dev set
    print("Preparing dev set with randomized positions...")
    dev_output = output_dir / "dev_set.csv"
    dev_fieldnames = [
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
    ]

    with open(dev_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dev_fieldnames)
        writer.writeheader()
        for row in dev_set:
            pos = assign_position(row, seed=row["prompt_id"])
            writer.writerow({
                "prompt_id": row["prompt_id"],
                "locale": row["locale"],
                "scenario_type": row["scenario_type"],
                "primary_criterion": row["primary_criterion"],
                "flaw_type": row["flaw_type"],
                "prompt_text": row["prompt_text"],
                "prompt_text_en": row["prompt_text_en"],
                "response_A": pos["response_A"],
                "response_A_en": pos["response_A_en"],
                "response_B": pos["response_B"],
                "response_B_en": pos["response_B_en"],
                "flipped": pos["flipped"],
            })
    print(f"  Output: {dev_output}")

    # Prepare test set with ground truth
    print("Preparing test set with randomized positions and ground truth...")
    test_output = output_dir / "test_set.csv"
    test_fieldnames = dev_fieldnames + ["ground_truth_winner"]

    with open(test_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames)
        writer.writeheader()
        for row in test_set:
            pos = assign_position(row, seed=row["prompt_id"])
            writer.writerow({
                "prompt_id": row["prompt_id"],
                "locale": row["locale"],
                "scenario_type": row["scenario_type"],
                "primary_criterion": row["primary_criterion"],
                "flaw_type": row["flaw_type"],
                "prompt_text": row["prompt_text"],
                "prompt_text_en": row["prompt_text_en"],
                "response_A": pos["response_A"],
                "response_A_en": pos["response_A_en"],
                "response_B": pos["response_B"],
                "response_B_en": pos["response_B_en"],
                "flipped": pos["flipped"],
                "ground_truth_winner": pos["ground_truth_winner"],
            })
    print(f"  Output: {test_output}")

    print("\nDataset preparation complete.")
    return dev_output, test_output


if __name__ == "__main__":
    input_csv = Path(__file__).parent.parent / "data" / "synthetic_prompts.csv"
    output_dir = Path(__file__).parent.parent / "data"

    dev_path, test_path = prepare_dataset(input_csv, output_dir, test_ratio=0.2, seed=42)
    print(f"\nOutput files:\n  {dev_path}\n  {test_path}")
