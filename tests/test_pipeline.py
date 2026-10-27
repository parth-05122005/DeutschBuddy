"""Offline tests for the RAG pipeline. No API key needed: the Gemini parts are replaced by fakes.

Run with:  python -m pytest tests  (or simply: python tests/test_pipeline.py)
"""

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import Settings  # noqa: E402
from rag.loader import chunk_markdown, load_directory, split_text  # noqa: E402
from rag.pipeline import DeutschBuddy  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashing embedder: similar words give similar vectors."""

    dim = 4096  # large enough that unrelated words rarely collide

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return (v / (np.linalg.norm(v) + 1e-12)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeChat:
    def __init__(self):
        self.last_prompt = ""

    def stream(self, prompt, system, history=None, temperature=0.3):
        self.last_prompt = prompt
        yield "Antwort "
        yield "[1]"

    def generate(self, prompt, system, temperature=0.3, json_schema=None):
        self.last_prompt = prompt
        if "questions" in json_schema["properties"]:
            return json.dumps({"questions": [
                {"question": "Ich fahre mit ___ Bus.", "options": ["der", "den", "dem", "des"],
                 "answer_index": 2, "explanation": "mit + Dativ", "source": 1},
                {"question": "broken", "options": ["a"], "answer_index": 5, "explanation": "", "source": 1},
            ]})
        return json.dumps({"german": "x", "english": "x", "literal": "x", "corrections": "",
                           "words": [], "grammar_points": []})


def make_engine(tmp: str) -> DeutschBuddy:
    cfg = Settings()
    cfg.db_dir = Path(tmp) / "db"
    cfg.min_score = 0.05
    return DeutschBuddy(cfg, embedder=FakeEmbedder(), chat=FakeChat())


def test_split_text_respects_size():
    text = "\n\n".join(f"Satz Nummer {i} ist hier." * 5 for i in range(30))
    pieces = split_text(text, size=300, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 300 + 50 + 2 for p in pieces)


def test_markdown_keeps_heading_path():
    md = "# Cases\n\nIntro.\n\n## Dativ\n\nmit, bei, zu take the Dativ."
    chunks = chunk_markdown(md, "cases.md", "grammar", 500, 50)
    assert chunks[-1].section == "Cases > Dativ"
    assert chunks[-1].text.startswith("[Cases > Dativ]")


def test_knowledge_base_loads():
    cfg = Settings()
    chunks = load_directory(cfg.knowledge_dir, cfg.chunk_size, cfg.chunk_overlap)
    assert {c.category for c in chunks} == {"grammar", "vocabulary", "culture"}
    assert len(chunks) > 50


def test_index_retrieve_and_features():
    # ChromaDB keeps its SQLite file open, so Windows may refuse to delete it immediately
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = make_engine(tmp)
        first = engine.index_knowledge_base()
        second = engine.index_knowledge_base()
        assert first.new_chunks == first.total_chunks > 0
        assert second.new_chunks == 0, "re-indexing must not duplicate chunks"

        hits = engine.retrieve("Oktoberfest Munich Theresienwiese beer")
        assert hits and hits[0].source == "festivals_and_holidays.md"

        grammar_only = engine.retrieve("Dativ prepositions mit bei", categories=["grammar"])
        assert all(h.category == "grammar" for h in grammar_only)

        hits, stream = engine.ask("What is Pfand?")
        assert "".join(stream) == "Antwort [1]"
        assert "[1]" in engine.chat.last_prompt

        questions, _ = engine.make_quiz("Dativ prepositions")
        assert len(questions) == 1, "malformed questions are dropped"

        report = engine.add_document("notes.txt", "Die Zugspitze ist der höchste Berg.".encode())
        assert report.new_chunks == 1
        assert "notes.txt" in engine.store.sources()
        engine.store.delete_source("notes.txt")
        assert "notes.txt" not in engine.store.sources()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
