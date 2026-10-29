"""Offline tests for the DaF-GQA benchmark and the experiment scripts. No API key needed.

Run with:  python tests/test_experiments.py
"""

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))

import common  # noqa: E402
import judge  # noqa: E402
import run as runner  # noqa: E402
import score  # noqa: E402
import sheets  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.loader import load_directory  # noqa: E402
from rag.pipeline import DeutschBuddy  # noqa: E402
from rag.prompts import ABSTAIN_MARKER  # noqa: E402
from test_pipeline import FakeEmbedder  # noqa: E402

# Table I of the paper: questions per topic and level
TABLE_I = {
    "articles": {"A1": 15, "A2": 10}, "cases": {"A1": 10, "A2": 10, "B1": 10},
    "verbs": {"A1": 15, "A2": 10, "B1": 5}, "word_order": {"A1": 10, "A2": 10, "B1": 5},
    "tenses": {"A1": 10, "A2": 10, "B1": 10}, "adjectives": {"A2": 10, "B1": 10},
}


class ScriptedChat:
    """Answers with a citation, or abstains when the abstention rule is in the system prompt."""

    model = "fake-model"

    def stream(self, prompt, system, history=None, temperature=0.3):
        if ABSTAIN_MARKER in system and "Konjunktiv" in prompt:
            yield ABSTAIN_MARKER + " Use würde + infinitive."
        elif "CONTEXT" in prompt:
            yield "Use the Dativ after mit [1]. "
            yield "Example: **Ich fahre mit dem Bus.** [9]"
        else:
            yield "Use the Dativ after mit."

    def generate(self, prompt, system, temperature=0.3, json_schema=None):
        return json.dumps({"correctness": "correct", "level_fit": "suitable", "reason": "Matches the reference.",
                           "claims": [{"claim": "mit takes the Dativ", "supported": True},
                                      {"claim": "Ich fahre mit dem Bus.", "supported": False}],
                           "citations": [{"passage": 1, "sentence": "Use the Dativ after mit", "supported": True},
                                         {"passage": 9, "sentence": "Example", "supported": True}]})


def make_engine(tmp: str) -> DeutschBuddy:
    cfg = Settings()
    cfg.db_dir = Path(tmp) / "db"
    cfg.min_score = 0.05
    engine = DeutschBuddy(cfg, embedder=FakeEmbedder(), chat=ScriptedChat())
    engine.index_knowledge_base()
    return engine


def test_benchmark_shape():
    questions = common.load_benchmark()
    assert len(questions) == 180
    assert len({q["id"] for q in questions}) == 180, "question ids must be unique"
    for q in questions:
        assert set(q) == {"id", "topic", "level", "lang", "question", "gold", "reference"}, q["id"]
        assert q["level"] in ("A1", "A2", "B1") and q["lang"] in ("en", "de", "mixed"), q["id"]
        assert q["question"].strip() and q["reference"].strip(), q["id"]
        assert bool(q["gold"]) == common.in_scope(q), f"{q['id']}: only out-of-scope questions lack gold labels"

    counts = Counter((q["topic"], q["level"]) for q in questions if common.in_scope(q))
    for topic, levels in TABLE_I.items():
        for level in ("A1", "A2", "B1"):
            assert counts[(topic, level)] == levels.get(level, 0), (topic, level, counts[(topic, level)])
    assert sum(not common.in_scope(q) for q in questions) == 20


def test_every_gold_label_matches_a_chunk():
    cfg = Settings()
    chunks = load_directory(cfg.knowledge_dir, cfg.chunk_size, cfg.chunk_overlap)
    for q in common.load_benchmark():
        for gold in q["gold"]:
            assert any(common.gold_match(gold, c.source, c.section) for c in chunks), f"{q['id']}: no chunk for {gold}"


def test_gold_match_rules():
    assert common.gold_match("cases.md#Always Dativ",
                             "cases.md", "The Four German Cases (Kasus) > Prepositions and their cases > Always Dativ")
    assert not common.gold_match("cases.md#Always Dativ", "cases.md", "The Four German Cases (Kasus) > Examples")
    # A title-level label matches only the title's own chunk, not every chunk of the file
    assert common.gold_match("cases.md#The Four German Cases (Kasus)", "cases.md", "The Four German Cases (Kasus)")
    assert not common.gold_match("cases.md#The Four German Cases (Kasus)", "cases.md",
                                 "The Four German Cases (Kasus) > Examples")
    retrieved = [{"source": "x.md", "section": "A > B"}, {"source": "cases.md", "section": "T > Always Dativ"}]
    assert common.hit_rank(["cases.md#Always Dativ"], retrieved) == 2


def test_run_judge_sheets_and_score():
    questions = [q for q in common.load_benchmark() if q["id"] in ("CAS-04", "CAS-12", "OOS-01")]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        results = Path(tmp) / "results"
        engine = make_engine(tmp)
        out = results / "answers.jsonl"

        assert runner.run_experiments(engine, questions, list(common.CONFIGS), out, log=lambda *_: None) == 12
        assert runner.run_experiments(engine, questions, list(common.CONFIGS), out, log=lambda *_: None) == 0
        answers = common.read_jsonl(out)
        assert len(answers) == 12
        by_key = {(r["id"], r["config"]): r for r in answers}
        assert by_key[("CAS-04", "C1")]["retrieved"] == [] and by_key[("CAS-04", "C1")]["hit_rank"] is None
        assert by_key[("CAS-04", "C2")]["retrieved"], "RAG configurations retrieve passages"
        assert by_key[("OOS-01", "C4")]["abstained"] and not by_key[("OOS-01", "C3")]["abstained"]

        judgments_path = results / "judgments.jsonl"
        references = {q["id"]: q["reference"] for q in questions}
        n = judge.run_judge(ScriptedChat(), answers, references, judgments_path, log=lambda *_: None)
        assert n == 8, "every in-scope answer is judged, closed-book included"
        judged = {(j["id"], j["config"]): j for j in common.read_jsonl(judgments_path)}
        closed, grounded = judged[("CAS-04", "C1")], judged[("CAS-04", "C2")]
        assert closed["correctness"] == "correct" and closed["faithfulness"] is None, "no passages, no faithfulness"
        assert grounded["faithfulness"] == 0.5
        assert grounded["citation_precision"] == 0.5, "a citation to a passage that does not exist is unsupported"
        assert judge.write_check_sample(answers, list(judged.values()), results / "sample.csv") >= 1

        # Judge labels and the teacher spot-check
        labels = score.judge_labels(list(judged.values()))
        sheets.make_spotcheck(answers, references, results, n=4)
        with (results / "spotcheck_teacher.csv").open(newline="", encoding="utf-8-sig") as f:
            spot = list(csv.DictReader(f))
        for r in spot:
            r["correctness"], r["level_fit"] = "correct", "suitable"
        with (results / "spotcheck_teacher.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=sheets.SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows(spot)
        agreement = score.spotcheck_agreement(results, labels)
        assert agreement["correctness"]["n"] == 4 and agreement["correctness"]["agreement"] == 1.0
        sheets.make_sheets(answers, references, results)
        with (results / "ratings_rater1.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12 and "config" not in rows[0], "sheets are blind to the configuration"

        # Fill both sheets: rater 2 disagrees on one item
        for rater, flip in ((1, None), (2, rows[0]["item_id"])):
            for r in rows:
                r["correctness"] = "incorrect" if r["item_id"] == flip else "correct"
                r["level_fit"] = "suitable"
            with (results / f"ratings_rater{rater}.csv").open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=sheets.SHEET_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

        labels, stats = score.load_ratings(results)
        assert stats["correctness"]["n"] == 12 and len(stats["correctness"]["disagreements"]) == 1
        tables = score.build_tables(answers, common.read_jsonl(judgments_path), labels, stats)
        assert "Table III" in tables and "Table IV" in tables and "Table V" in tables
        assert "| C4 RAG + level + abstain | 100.0 |" in tables, "C4 flags the out-of-scope question"


def test_statistics():
    assert score.cohen_kappa(["x", "x", "y", "y"], ["x", "x", "y", "x"]) == 0.5
    assert score.cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    b, c, p = score.mcnemar_exact([False] * 5 + [True] * 3, [True] * 5 + [True] * 3)
    assert (b, c) == (0, 5) and abs(p - 0.0625) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    langs = Counter(q["lang"] for q in common.load_benchmark())
    print("question languages:", dict(langs))
