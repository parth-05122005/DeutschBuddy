"""Step 3: build blind rating sheets for the two human raters.

Run:  python experiments/sheets.py
Writes ratings_rater1.csv and ratings_rater2.csv (identical, shuffled, without the configuration)
and rating_key.csv (which item is which question and configuration). Raters fill the columns
correctness (correct / partly / incorrect) and level_fit (suitable / too_difficult / too_simple).
Keep rating_key.csv away from the raters.
"""

import argparse
import csv
import random
from pathlib import Path

from common import RESULTS_DIR, load_benchmark, read_jsonl

SHEET_COLUMNS = ["item_id", "level", "question", "reference_answer", "answer", "correctness", "level_fit", "comment"]
SEED = 42


def make_sheets(answers: list[dict], references: dict[str, str], out_dir: Path, raters: int = 2,
                seed: int = SEED) -> list[Path]:
    items = sorted(answers, key=lambda r: (r["id"], r["config"]))
    random.Random(seed).shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so that Excel shows German characters correctly
    with (out_dir / "rating_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "id", "config"])
        for n, r in enumerate(items, start=1):
            writer.writerow([f"R{n:03d}", r["id"], r["config"]])

    paths = []
    for rater in range(1, raters + 1):
        path = out_dir / f"ratings_rater{rater}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            for n, r in enumerate(items, start=1):
                writer.writerow({"item_id": f"R{n:03d}", "level": r["level"], "question": r["question"],
                                 "reference_answer": references[r["id"]], "answer": r["answer"],
                                 "correctness": "", "level_fit": "", "comment": ""})
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, default=RESULTS_DIR / "answers.jsonl")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--raters", type=int, default=2)
    args = parser.parse_args()

    references = {q["id"]: q["reference"] for q in load_benchmark()}
    paths = make_sheets(read_jsonl(args.answers), references, args.out_dir, args.raters)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
