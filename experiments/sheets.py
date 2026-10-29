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


def make_spotcheck(answers: list[dict], references: dict[str, str], out_dir: Path, n: int = 50,
                   seed: int = SEED) -> Path:
    """A blind random sample of in-scope answers for a teacher to rate, to measure how well the
    LLM judge agrees with a human. Writes spotcheck_teacher.csv and spotcheck_key.csv."""
    pool = sorted((r for r in answers if r["in_scope"]), key=lambda r: (r["id"], r["config"]))
    sample = random.Random(seed).sample(pool, k=min(n, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "spotcheck_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "id", "config"])
        for k, r in enumerate(sample, start=1):
            writer.writerow([f"S{k:02d}", r["id"], r["config"]])
    path = out_dir / "spotcheck_teacher.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for k, r in enumerate(sample, start=1):
            writer.writerow({"item_id": f"S{k:02d}", "level": r["level"], "question": r["question"],
                             "reference_answer": references[r["id"]], "answer": r["answer"],
                             "correctness": "", "level_fit": "", "comment": ""})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, default=RESULTS_DIR / "answers.jsonl")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--raters", type=int, default=2)
    parser.add_argument("--spotcheck", type=int, metavar="N",
                        help="instead of full rating sheets, write a blind sample of N answers for a teacher")
    args = parser.parse_args()

    references = {q["id"]: q["reference"] for q in load_benchmark()}
    answers = read_jsonl(args.answers)
    if args.spotcheck:
        print(f"Wrote {make_spotcheck(answers, references, args.out_dir, args.spotcheck)}")
        return
    for p in make_sheets(answers, references, args.out_dir, args.raters):
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
