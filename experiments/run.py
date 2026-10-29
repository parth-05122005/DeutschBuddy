"""Step 1: answer every benchmark question with every configuration and save the results.

Run:  python experiments/run.py --delay 7
The run is resumable: answers already in the output file are skipped, so after a quota error
simply start it again.
"""

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from common import CONFIGS, RESULTS_DIR, append_jsonl, hit_rank, in_scope, load_benchmark, read_jsonl

from rag import DeutschBuddy
from rag.config import settings
from rag.prompts import ABSTAIN_MARKER

TEMPERATURE = 0.2


# Without the abstention rule (C2, C3) the model may still decline in its own words
REFUSAL = re.compile(
    r"(passages?|context|knowledge base|notes)[^.]{0,60}(do(es)? not|don't|doesn't) (contain|cover|include|mention)"
    r"|not (covered|included|mentioned) in the (provided )?(passages|context)"
    r"|(passagen|kontext)[^.]{0,60}(enthalten|enthält) keine", re.IGNORECASE)


def is_abstention(answer: str) -> bool:
    opening = answer[:300]
    return ABSTAIN_MARKER.lower() in opening.lower() or bool(REFUSAL.search(opening))


def answer_one(engine, question: dict, config_id: str) -> dict:
    flags = {k: v for k, v in CONFIGS[config_id].items() if k != "name"}
    start = time.perf_counter()
    hits, stream = engine.ask(question["question"], history=None, level=question["level"], language="English",
                              categories=["grammar"], temperature=TEMPERATURE, **flags)
    first_token, parts = None, []
    for part in stream:
        if first_token is None:
            first_token = time.perf_counter() - start
        parts.append(part)
    answer = "".join(parts)
    retrieved = [{"source": h.source, "section": h.section, "score": round(h.score, 4), "text": h.text}
                 for h in hits]
    return {
        "id": question["id"], "config": config_id, "topic": question["topic"], "level": question["level"],
        "lang": question["lang"], "in_scope": in_scope(question), "question": question["question"],
        "answer": answer, "retrieved": retrieved,
        "hit_rank": hit_rank(question["gold"], retrieved) if retrieved else None,
        "abstained": is_abstention(answer),
        "latency_first_s": round(first_token or 0.0, 3), "latency_total_s": round(time.perf_counter() - start, 3),
        "model": getattr(engine.chat, "model", "unknown"), "temperature": TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_experiments(engine, questions: list[dict], configs: list[str], out_path: Path,
                    delay: float = 0.0, log=print) -> int:
    """Answer each (question, config) pair not yet in out_path. Returns the number of new answers."""
    done = {(r["id"], r["config"]) for r in read_jsonl(out_path)}
    todo = [(q, c) for q in questions for c in configs if (q["id"], c) not in done]
    log(f"{len(done)} answers already saved, {len(todo)} to go")
    failures_in_a_row = 0
    for i, (question, config_id) in enumerate(todo, start=1):
        try:
            record = answer_one(engine, question, config_id)
            failures_in_a_row = 0
        except Exception as exc:  # keep going; the pair is retried on the next run
            log(f"[{i}/{len(todo)}] {question['id']} {config_id} FAILED: {str(exc)[:300]}")
            failures_in_a_row += 1
            if "PerDay" in str(exc):
                log("Daily quota reached. Run the same command again tomorrow to continue.")
                break
            if failures_in_a_row >= 5:
                log("5 failures in a row (network down?). Stopping; run again to continue.")
                break
            continue
        append_jsonl(out_path, record)
        log(f"[{i}/{len(todo)}] {question['id']} {config_id} ok ({record['latency_total_s']} s)")
        if delay:
            time.sleep(delay)
    return len(todo)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--ids", nargs="+", help="only these question ids (for a quick trial)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--delay", type=float, default=7.0,
                        help="seconds to wait between calls (free tier: about 10 requests per minute)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "answers.jsonl")
    args = parser.parse_args()

    questions = load_benchmark()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[:args.limit]

    engine = DeutschBuddy()
    engine.index_knowledge_base()
    print(f"Model: {settings.chat_model} · embeddings: {settings.embed_model} · top-k {settings.top_k} "
          f"· min score {settings.min_score}")
    run_experiments(engine, questions, args.configs, args.out, args.delay)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
