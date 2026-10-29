"""Step 2: automatic evaluation with an LLM judge.

Run:  python experiments/judge.py --delay 4
For every in-scope answer (all four configurations) the judge, in a single call:
  - rates correctness against the teacher-checked reference answer (correct / partly / incorrect),
  - rates level fit for the question's CEFR level (suitable / too_difficult / too_simple),
and for answers that were given retrieved passages (C2-C4) also:
  - splits the answer into factual claims and checks each against the passages
    (faithfulness, following the RAGAS definition),
  - checks every inline citation [n] against passage n (citation precision).
It also writes judge_check_sample.csv: a 20% sample of claim and citation verdicts for a human
to verify. Resumable like run.py. JUDGE_MODEL in .env selects the judge (default Gemini 3.1 Flash-Lite,
a different model from the one being evaluated).
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

from common import RESULTS_DIR, append_jsonl, load_benchmark, read_jsonl

DEFAULT_JUDGE = "gemini-3.1-flash-lite"

JUDGE_SYSTEM = """You are a strict evaluator of answers from a German grammar tutor for learners.
You receive a learner QUESTION, its CEFR LEVEL, a REFERENCE answer checked by a German teacher,
the tutor's ANSWER, and possibly numbered PASSAGES the tutor was given.

1. correctness: compare the ANSWER with the REFERENCE.
   - "correct": the answer gives the same key point as the reference and states nothing wrong
     about German.
   - "partly": the key point is right, but the answer is incomplete or contains a minor error.
   - "incorrect": the key point is wrong or missing, or the answer contains a major error about
     German grammar. An answer that only says it cannot help is "incorrect".
   Extra correct detail beyond the reference is fine. The answer may be written in German.
2. level_fit: is the explanation suitable for a learner at the given CEFR level (vocabulary,
   sentence length, grammatical terminology, amount of detail)? "suitable", "too_difficult" or
   "too_simple".
3. claims: only if PASSAGES are given. Split the ANSWER into its individual factual statements
   about German (rules, forms, examples, translations); ignore greetings and encouragement. For
   each, decide whether it can be directly inferred from the PASSAGES. A claim that is true in
   general but not stated in the passages is NOT supported. With no passages, return [].
4. citations: only if PASSAGES are given. For every citation marker such as [2] in the ANSWER,
   give the passage number and the sentence it is attached to, and decide whether that passage
   supports that sentence. With no passages or no markers, return []."""

JUDGE_PROMPT = """QUESTION: {question}
CEFR LEVEL: {level}

REFERENCE:
{reference}

PASSAGES:
{passages}

ANSWER:
{answer}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "partly", "incorrect"]},
        "level_fit": {"type": "string", "enum": ["suitable", "too_difficult", "too_simple"]},
        "reason": {"type": "string", "description": "One sentence explaining the correctness rating"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["claim", "supported"]}},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "passage": {"type": "integer"}, "sentence": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["passage", "sentence", "supported"]}},
    },
    "required": ["correctness", "level_fit", "reason", "claims", "citations"],
}

CITATION = re.compile(r"\[(\d+)\]")


def parse_json(raw: str) -> dict:
    """Read the first JSON object in the reply; some models add a code fence or trailing text."""
    text = raw.strip().removeprefix("```json").removeprefix("```").strip()
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def format_passages(retrieved: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(retrieved, start=1)) or "(none)"


def score_judgment(verdict: dict, n_passages: int) -> dict:
    claims = verdict.get("claims", []) if n_passages else []
    # A citation to a passage number that does not exist can never be supported
    citations = [dict(c, supported=c["supported"] and 1 <= c["passage"] <= n_passages)
                 for c in (verdict.get("citations", []) if n_passages else [])]
    return {
        "correctness": verdict.get("correctness"), "level_fit": verdict.get("level_fit"),
        "reason": verdict.get("reason", ""),
        "n_claims": len(claims),
        "faithfulness": round(sum(c["supported"] for c in claims) / len(claims), 4) if claims else None,
        "n_citations": len(citations),
        "citation_precision": round(sum(c["supported"] for c in citations) / len(citations), 4) if citations else None,
        "claims": claims, "citations": citations,
    }


def judge_one(chat, record: dict, reference: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=record["question"], level=record["level"], reference=reference,
                                 passages=format_passages(record["retrieved"]), answer=record["answer"])
    raw = chat.generate(prompt, JUDGE_SYSTEM, temperature=0.0, json_schema=JUDGE_SCHEMA)
    result = score_judgment(parse_json(raw), len(record["retrieved"]))
    result["markers_in_answer"] = len(CITATION.findall(record["answer"]))
    return {"id": record["id"], "config": record["config"], "judge": getattr(chat, "model", "unknown"), **result}


def needs_judging(record: dict) -> bool:
    return record["in_scope"]


def run_judge(chat, answers: list[dict], references: dict[str, str], out_path: Path,
              delay: float = 0.0, log=print) -> int:
    done = {(r["id"], r["config"]) for r in read_jsonl(out_path)}
    todo = [r for r in answers if needs_judging(r) and (r["id"], r["config"]) not in done]
    log(f"{len(done)} judgments already saved, {len(todo)} to go")
    failures_in_a_row = 0
    for i, record in enumerate(todo, start=1):
        try:
            append_jsonl(out_path, judge_one(chat, record, references[record["id"]]))
            log(f"[{i}/{len(todo)}] {record['id']} {record['config']} judged")
            failures_in_a_row = 0
        except Exception as exc:
            log(f"[{i}/{len(todo)}] {record['id']} {record['config']} FAILED: {str(exc)[:300]}")
            failures_in_a_row += 1
            if "PerDay" in str(exc):
                log("Daily quota reached. Run the same command again tomorrow to continue.")
                break
            if failures_in_a_row >= 5:
                log("5 failures in a row (network down?). Stopping; run again to continue.")
                break
        if delay:
            time.sleep(delay)
    return len(todo)


def write_check_sample(answers: list[dict], judgments: list[dict], path: Path, share: float = 0.2,
                       seed: int = 42) -> int:
    """A random sample of grounded answers for a human to confirm or reject each claim and citation verdict."""
    by_key = {(r["id"], r["config"]): r for r in answers}
    grounded = [j for j in judgments if by_key.get((j["id"], j["config"]), {}).get("retrieved")]
    sample = random.Random(seed).sample(grounded, k=max(1, round(len(grounded) * share))) if grounded else []
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
    parser.add_argument("--delay", type=float, default=4.0)
    args = parser.parse_args()

    from google import genai
    from google.genai import types
    from rag.config import settings
    from rag.gemini import GeminiChat

    model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE)
    # A request that never returns would otherwise stall the whole run
    client = genai.Client(api_key=settings.api_key, http_options=types.HttpOptions(timeout=90_000))
    chat = GeminiChat(client, model)
    print(f"Judge model: {model}")
    answers = read_jsonl(args.answers)
    references = {q["id"]: q["reference"] for q in load_benchmark()}
    run_judge(chat, answers, references, args.out, args.delay)
    n = write_check_sample(answers, read_jsonl(args.out), args.out.parent / "judge_check_sample.csv")
    print(f"Saved to {args.out}; {n} answers in judge_check_sample.csv for human checking")


if __name__ == "__main__":
    main()
