"""Step 4: compute the numbers for the paper (Tables III-V) from answers, judgments and ratings.

Run:  python experiments/score.py
Optional: ratings_final.csv (same layout as the rater sheets) with the labels agreed after
discussing disagreements. Without it, rater 1's labels are used and the disagreements are listed.
Writes results/tables.md.
"""

import argparse
import csv
from math import comb
from pathlib import Path
from statistics import mean

from common import CONFIGS, RESULTS_DIR, read_jsonl

CORRECTNESS = {"correct": "correct", "c": "correct", "partly": "partly", "p": "partly", "partly correct": "partly",
               "incorrect": "incorrect", "i": "incorrect", "wrong": "incorrect"}
LEVEL_FIT = {"suitable": "suitable", "s": "suitable", "too_difficult": "too_difficult", "too difficult": "too_difficult",
             "d": "too_difficult", "too_simple": "too_simple", "too simple": "too_simple", "e": "too_simple"}


# ---------- Statistics ----------

def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters' labels on the same items."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def mcnemar_exact(first: list[bool], second: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired binary outcomes. Returns (b, c, p)."""
    b = sum(x and not y for x, y in zip(first, second))
    c = sum(y and not x for x, y in zip(first, second))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, p)


# ---------- Loading ratings ----------

def read_sheet(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["item_id"]: row for row in csv.DictReader(f)}


def normalise(value: str, mapping: dict[str, str]) -> str | None:
    return mapping.get((value or "").strip().lower())


def load_ratings(results: Path) -> tuple[dict, dict]:
    """Returns ({(id, config): {"correctness", "level_fit"}}, agreement statistics)."""
    with (results / "rating_key.csv").open(newline="", encoding="utf-8-sig") as f:
        key = {row["item_id"]: (row["id"], row["config"]) for row in csv.DictReader(f)}
    r1, r2 = read_sheet(results / "ratings_rater1.csv"), read_sheet(results / "ratings_rater2.csv")
    final = read_sheet(results / "ratings_final.csv")

    stats = {}
    for field, mapping in (("correctness", CORRECTNESS), ("level_fit", LEVEL_FIT)):
        pairs = [(normalise(r1[i][field], mapping), normalise(r2[i][field], mapping))
                 for i in key if i in r1 and i in r2]
        pairs = [(x, y) for x, y in pairs if x and y]
        stats[field] = {"n": len(pairs), "kappa": cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]),
                        "disagreements": sorted(i for i in key if i in r1 and i in r2
                                                and normalise(r1[i][field], mapping) != normalise(r2[i][field], mapping))}

    labels = {}
    for item, pair in key.items():
        source = final.get(item) or r1.get(item) or {}
        labels[pair] = {"correctness": normalise(source.get("correctness", ""), CORRECTNESS),
                        "level_fit": normalise(source.get("level_fit", ""), LEVEL_FIT)}
    stats["used_final"] = bool(final)
    return labels, stats


def judge_labels(judgments: list[dict]) -> dict:
    """Correctness and level fit as rated by the LLM judge."""
    return {(j["id"], j["config"]): {"correctness": normalise(j.get("correctness", ""), CORRECTNESS),
                                     "level_fit": normalise(j.get("level_fit", ""), LEVEL_FIT)}
            for j in judgments}


def spotcheck_agreement(results: Path, labels: dict) -> dict:
    """Agreement between the judge and a teacher on the blind spot-check sample, if it is filled in."""
    key_path, sheet = results / "spotcheck_key.csv", read_sheet(results / "spotcheck_teacher.csv")
    if not key_path.exists() or not sheet:
        return {}
    with key_path.open(newline="", encoding="utf-8-sig") as f:
        key = {row["item_id"]: (row["id"], row["config"]) for row in csv.DictReader(f)}
    stats = {}
    for field, mapping in (("correctness", CORRECTNESS), ("level_fit", LEVEL_FIT)):
        pairs = [(normalise(sheet[i][field], mapping), labels.get(pair, {}).get(field))
                 for i, pair in key.items() if i in sheet]
        pairs = [(h, j) for h, j in pairs if h and j]
        stats[field] = {"n": len(pairs), "kappa": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs]),
                        "agreement": sum(h == j for h, j in pairs) / len(pairs) if pairs else None}
    return stats


# ---------- Tables ----------

def pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.1f}" if values else "n/a"


def avg(values: list[float]) -> str:
    values = [v for v in values if v is not None]
    return f"{mean(values):.3f}" if values else "n/a"


def build_tables(answers: list[dict], judgments: list[dict], labels: dict, stats: dict) -> str:
    judged = {(j["id"], j["config"]): j for j in judgments}
    configs = [c for c in CONFIGS if any(r["config"] == c for r in answers)]
    out = []

    # Table III: correctness, faithfulness, citations, level fit (in-scope questions)
    out += ["## Table III. Results on in-scope questions", "",
            "| Config. | n | Correct (%) | Partly (%) | Incorrect (%) | Faithfulness | Citation precision | Level fit (%) | First token (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    correct_by_config = {}
    for c in configs:
        rows = [r for r in answers if r["config"] == c and r["in_scope"]]
        lab = [labels.get((r["id"], c), {}) for r in rows]
        corr = [x["correctness"] for x in lab if x.get("correctness")]
        fit = [x["level_fit"] for x in lab if x.get("level_fit")]
        correct_by_config[c] = {r["id"]: labels.get((r["id"], c), {}).get("correctness") == "correct" for r in rows}
        js = [judged[(r["id"], c)] for r in rows if (r["id"], c) in judged]
        out.append(f"| {c} {CONFIGS[c]['name']} | {len(rows)} | {pct([x == 'correct' for x in corr])} | "
                   f"{pct([x == 'partly' for x in corr])} | {pct([x == 'incorrect' for x in corr])} | "
                   f"{avg([j['faithfulness'] for j in js]) if c != 'C1' else '–'} | "
                   f"{avg([j['citation_precision'] for j in js]) if c != 'C1' else '–'} | "
                   f"{pct([x == 'suitable' for x in fit])} | {avg([r['latency_first_s'] for r in rows])} |")
    k = stats.get("correctness", {}), stats.get("level_fit", {})
    if stats.get("source") == "judge":
        line = "Correctness and level fit rated by the LLM judge."
        if k[0].get("n"):
            line += (f" Judge–teacher agreement on {k[0]['n']} spot-checked answers: correctness "
                     f"κ = {fmt(k[0]['kappa'])} ({100 * k[0]['agreement']:.0f}% identical), level fit "
                     f"κ = {fmt(k[1].get('kappa'))} ({100 * (k[1].get('agreement') or 0):.0f}% identical).")
        else:
            line += " Teacher spot-check not filled in yet."
        out += ["", line, ""]
    else:
        out += ["", f"Inter-rater agreement: correctness κ = {fmt(k[0].get('kappa'))} (n = {k[0].get('n', 0)}), "
                    f"level fit κ = {fmt(k[1].get('kappa'))} (n = {k[1].get('n', 0)}).", ""]

    # McNemar tests on 'fully correct' between neighbouring configurations
    out += ["### Paired tests (fully correct, McNemar exact)", "", "| Comparison | b | c | p |", "| --- | --- | --- | --- |"]
    for x, y in (("C1", "C2"), ("C2", "C3"), ("C3", "C4")):
        if x in correct_by_config and y in correct_by_config:
            ids = sorted(set(correct_by_config[x]) & set(correct_by_config[y]))
            b, c_, p = mcnemar_exact([correct_by_config[x][i] for i in ids], [correct_by_config[y][i] for i in ids])
            out.append(f"| {x} vs {y} | {b} | {c_} | {p:.4f} |")
    out.append("")

    # Correctness and level fit by CEFR level (RQ2 detail)
    out += ["### By CEFR level", "", "| Config. | Level | n | Correct (%) | Level fit (%) | Too difficult (%) |",
            "| --- | --- | --- | --- | --- | --- |"]
    for c in configs:
        for level in ("A1", "A2", "B1"):
            lab = [labels.get((r["id"], c), {}) for r in answers if r["config"] == c and r["in_scope"] and r["level"] == level]
            fit = [x.get("level_fit") for x in lab if x.get("level_fit")]
            corr = [x.get("correctness") for x in lab if x.get("correctness")]
            out.append(f"| {c} | {level} | {len(lab)} | {pct([x == 'correct' for x in corr])} | "
                       f"{pct([x == 'suitable' for x in fit])} | {pct([x == 'too_difficult' for x in fit])} |")
    out.append("")

    # Table IV: retrieval quality by question language (retrieval is identical in C2-C4, so use C2)
    rag_rows = [r for r in answers if r["config"] == "C2" and r["in_scope"]]
    out += ["## Table IV. Retrieval quality by question language", "", "| Question language | n | Hit@5 | MRR |",
            "| --- | --- | --- | --- |"]
    for name, lang in (("English", "en"), ("German", "de"), ("Mixed", "mixed"), ("All in-scope", None)):
        rows = [r for r in rag_rows if lang is None or r["lang"] == lang]
        out.append(f"| {name} | {len(rows)} | {avg([1.0 if r['hit_rank'] else 0.0 for r in rows])} | "
                   f"{avg([1 / r['hit_rank'] if r['hit_rank'] else 0.0 for r in rows])} |")
    out.append("")

    # Table V: abstention
    out += ["## Table V. Abstention", "",
            "| Config. | Out-of-scope flagged (%) | Precision | False abstentions on in-scope (%) |",
            "| --- | --- | --- | --- |"]
    for c in configs:
        rows = [r for r in answers if r["config"] == c]
        oos = [r["abstained"] for r in rows if not r["in_scope"]]
        ins = [r["abstained"] for r in rows if r["in_scope"]]
        flagged = sum(oos) + sum(ins)
        precision = f"{sum(oos) / flagged:.3f}" if flagged else "n/a"
        out.append(f"| {c} {CONFIGS[c]['name']} | {pct(oos)} | {precision} | {pct(ins)} |")
    out.append("")

    if stats.get("correctness", {}).get("disagreements") and not stats.get("used_final"):
        out += [f"Note: {len(stats['correctness']['disagreements'])} correctness disagreements between raters. "
                "Resolve them in ratings_final.csv; until then rater 1's labels are used.", ""]
    return "\n".join(out)


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    answers = read_jsonl(args.results / "answers.jsonl")
    judgments = read_jsonl(args.results / "judgments.jsonl")
    human = (args.results / "rating_key.csv").exists() and (args.results / "ratings_rater1.csv").exists()
    if human and any(read_sheet(args.results / "ratings_rater1.csv")[i]["correctness"]
                     for i in read_sheet(args.results / "ratings_rater1.csv")):
        labels, stats = load_ratings(args.results)
    else:
        # No human ratings: use the LLM judge, validated against the teacher's spot-check
        labels = judge_labels(judgments)
        stats = {"source": "judge", **spotcheck_agreement(args.results, labels)}
    tables = build_tables(answers, judgments, labels, stats)
    (args.results / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)


if __name__ == "__main__":
    main()
