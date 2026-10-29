"""Step 2: automatic faithfulness and citation precision with an LLM judge.

Run:  python experiments/judge.py --delay 7
For every grounded, in-scope answer (C2-C4) the judge splits the answer into factual claims
and checks each against the retrieved passages (faithfulness, following the RAGAS definition),
and checks every inline citation [n] against passage n (citation precision).
It also writes judge_check_sample.csv: a 20% random sample for a human to verify the judge.
Resumable like run.py. Set JUDGE_MODEL in .env to use a different model from the one being judged.
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

from common import RESULTS_DIR, append_jsonl, read_jsonl

JUDGE_SYSTEM = """You are a strict evaluator of answers from a German grammar tutor.
You receive numbered PASSAGES and an ANSWER that should be based on them.

1. claims: split the ANSWER into its individual factual statements about German (rules, forms,
   examples, translations). Ignore greetings and encouragement. For each claim decide whether it
   can be directly inferred from the PASSAGES (supported = true) or not (false). A claim that is
   true in general but not stated in the passages is NOT supported.
2. citations: for every citation marker such as [2] in the ANSWER, give the passage number and the
   sentence it is attached to, and decide whether that passage supports that sentence."""

JUDGE_PROMPT = """PASSAGES:
{passages}

ANSWER:
{answer}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["claim", "supported"]}},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "passage": {"type": "integer"}, "sentence": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["passage", "sentence", "supported"]}},
    },
    "required": ["claims", "citations"],
}

CITATION = re.compile(r"\[(\d+)\]")


def format_passages(retrieved: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(retrieved, start=1)) or "(none)"


def score_judgment(verdict: dict, n_passages: int) -> dict:
    claims = verdict.get("claims", [])
    # A citation to a passage number that does not exist can never be supported
    citations = [dict(c, supported=c["supported"] and 1 <= c["passage"] <= n_passages)
                 for c in verdict.get("citations", [])]
    return {
        "n_claims": len(claims),
        "faithfulness": round(sum(c["supported"] for c in claims) / len(claims), 4) if claims else None,
        "n_citations": len(citations),
        "citation_precision": round(sum(c["supported"] for c in citations) / len(citations), 4) if citations else None,
        "claims": claims, "citations": citations,
    }


def judge_one(chat, record: dict) -> dict:
    raw = chat.generate(JUDGE_PROMPT.format(passages=format_passages(record["retrieved"]), answer=record["answer"]),
                        JUDGE_SYSTEM, temperature=0.0, json_schema=JUDGE_SCHEMA)
    result = score_judgment(json.loads(raw), len(record["retrieved"]))
    result["markers_in_answer"] = len(CITATION.findall(record["answer"]))
    return {"id": record["id"], "config": record["config"], **result}


def needs_judging(record: dict) -> bool:
    return record["config"] != "C1" and record["in_scope"] and bool(record["retrieved"])


def run_judge(chat, answers: list[dict], out_path: Path, delay: float = 0.0, log=print) -> int:
    done = {(r["id"], r["config"]) for r in read_jsonl(out_path)}
    todo = [r for r in answers if needs_judging(r) and (r["id"], r["config"]) not in done]
    log(f"{len(done)} judgments already saved, {len(todo)} to go")
    for i, record in enumerate(todo, start=1):
        try:
            append_jsonl(out_path, judge_one(chat, record))
            log(f"[{i}/{len(todo)}] {record['id']} {record['config']} judged")
        except Exception as exc:
            log(f"[{i}/{len(todo)}] {record['id']} {record['config']} FAILED: {exc}")
        if delay:
            time.sleep(delay)
    return len(todo)


def write_check_sample(answers: list[dict], judgments: list[dict], path: Path, share: float = 0.2,
                       seed: int = 42) -> int:
    """A random sample of judged answers for a human to confirm or reject each judge verdict."""
    by_key = {(r["id"], r["config"]): r for r in answers}
    sample = random.Random(seed).sample(judgments, k=max(1, round(len(judgments) * share))) if judgments else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "config", "kind", "passage", "text", "judge_supported", "human_agrees"])
        for j in sample:
            record = by_key[(j["id"], j["config"])]
            writer.writerow([j["id"], j["config"], "ANSWER", "", record["answer"], "", ""])
            for n, chunk in enumerate(record["retrieved"], start=1):
                writer.writerow([j["id"], j["config"], "PASSAGE", n, chunk["text"], "", ""])
            for c in j["claims"]:
                writer.writerow([j["id"], j["config"], "claim", "", c["claim"], c["supported"], ""])
            for c in j["citations"]:
                writer.writerow([j["id"], j["config"], "citation", c["passage"], c["sentence"], c["supported"], ""])
    return len(sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, default=RESULTS_DIR / "answers.jsonl")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "judgments.jsonl")
    parser.add_argument("--delay", type=float, default=7.0)
    args = parser.parse_args()

    from google import genai
    from rag.config import settings
    from rag.gemini import GeminiChat

    model = os.getenv("JUDGE_MODEL", settings.chat_model)
    chat = GeminiChat(genai.Client(api_key=settings.api_key), model)
    print(f"Judge model: {model}")
    answers = read_jsonl(args.answers)
    run_judge(chat, answers, args.out, args.delay)
    n = write_check_sample(answers, read_jsonl(args.out), args.out.parent / "judge_check_sample.csv")
    print(f"Saved to {args.out}; {n} answers in judge_check_sample.csv for human checking")


if __name__ == "__main__":
    main()
