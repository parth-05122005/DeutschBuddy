"""Shared helpers: benchmark loading, configurations, gold-section matching, JSONL files."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK = PROJECT_ROOT / "benchmark" / "dafgqa.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# The four configurations of the paper (Table II)
CONFIGS = {
    "C1": {"name": "Closed-book", "use_retrieval": False, "level_control": False, "abstain": False},
    "C2": {"name": "RAG", "use_retrieval": True, "level_control": False, "abstain": False},
    "C3": {"name": "RAG + level", "use_retrieval": True, "level_control": True, "abstain": False},
    "C4": {"name": "RAG + level + abstain", "use_retrieval": True, "level_control": True, "abstain": True},
}

OUT_OF_SCOPE = "out_of_scope"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_benchmark(path: Path = BENCHMARK) -> list[dict]:
    return read_jsonl(path)


def in_scope(question: dict) -> bool:
    return question["topic"] != OUT_OF_SCOPE


def gold_match(gold: str, source: str, section: str) -> bool:
    """A gold label is "file.md#Heading". A chunk matches when it comes from that file and
    lies under that heading: the heading is part of the chunk's section path below the title,
    or the chunk is the title section itself."""
    file, heading = gold.split("#", 1)
    if source != file:
        return False
    parts = section.split(" > ")
    return heading in parts[1:] or parts == [heading]


def hit_rank(gold: list[str], retrieved: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk that matches any gold label, or None."""
    for rank, chunk in enumerate(retrieved, start=1):
        if any(gold_match(g, chunk["source"], chunk["section"]) for g in gold):
            return rank
    return None
