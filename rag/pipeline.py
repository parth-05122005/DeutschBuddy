"""The RAG pipeline: index documents, retrieve relevant chunks, generate grounded answers."""

import json
import random
from dataclasses import dataclass
from typing import Iterator

from google import genai

from . import prompts
from .config import Settings, settings as default_settings
from .gemini import GeminiChat, GeminiEmbedder
from .loader import Chunk, load_directory, load_file
from .vectorstore import SearchHit, VectorStore


@dataclass
class IndexReport:
    total_chunks: int
    new_chunks: int


class DeutschBuddy:
    def __init__(self, cfg: Settings | None = None, embedder=None, chat=None):
        self.cfg = cfg or default_settings
        if embedder is None or chat is None:
            if not self.cfg.api_key:
                raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY in the .env file.")
            client = genai.Client(api_key=self.cfg.api_key)
            embedder = embedder or GeminiEmbedder(client, self.cfg.embed_model, self.cfg.embed_dim)
            chat = chat or GeminiChat(client, self.cfg.chat_model)
        self.embedder = embedder
        self.chat = chat
        self.store = VectorStore(self.cfg.db_dir, self.cfg.collection)

    # ---------- Indexing ----------

    def _index(self, chunks: list[Chunk]) -> IndexReport:
        # Only embed chunks that are not stored yet (saves API calls on restart)
        known = self.store.existing_ids([c.id for c in chunks])
        new = [c for c in chunks if c.id not in known]
        if new:
            self.store.add(new, self.embedder.embed_documents([c.text for c in new]))
        return IndexReport(total_chunks=len(chunks), new_chunks=len(new))

    def index_knowledge_base(self) -> IndexReport:
        chunks = load_directory(self.cfg.knowledge_dir, self.cfg.chunk_size, self.cfg.chunk_overlap)
        return self._index(chunks)

    def add_document(self, name: str, data: bytes) -> IndexReport:
        chunks = load_file(name, data, "uploads", self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            raise ValueError(f"No text could be extracted from {name}. Scanned PDFs need OCR first.")
        return self._index(chunks)

    # ---------- Retrieval ----------

    def retrieve(self, query: str, k: int | None = None, categories: list[str] | None = None) -> list[SearchHit]:
        hits = self.store.search(self.embedder.embed_query(query), k or self.cfg.top_k, categories)
        return [h for h in hits if h.score >= self.cfg.min_score]

    @staticmethod
    def format_context(hits: list[SearchHit]) -> str:
        if not hits:
            return "(no relevant passages found)"
        blocks = []
        for i, hit in enumerate(hits, start=1):
            where = f"{hit.source}" + (f", page {hit.page}" if hit.page else "")
            blocks.append(f"[{i}] ({where})\n{hit.text}")
        return "\n\n".join(blocks)

    # ---------- Features ----------

    def ask(self, question: str, history: list[dict] | None = None, level: str = "A2",
            language: str = "English", categories: list[str] | None = None, use_retrieval: bool = True,
            level_control: bool = True, abstain: bool = True,
            temperature: float = 0.3) -> tuple[list[SearchHit], Iterator[str]]:
        """Answer a question. Returns the sources first, then a token stream of the answer.

        With use_retrieval=False the model answers closed-book (no passages in the prompt).
        """
        hits: list[SearchHit] = []
        if use_retrieval:
            # Follow-up questions like "and in the Dativ?" need the previous question to retrieve well
            last_user = next((t["content"] for t in reversed(history or []) if t["role"] == "user"), "")
            search_query = f"{last_user}\n{question}" if last_user and len(question) < 60 else question
            hits = self.retrieve(search_query, categories=categories)
            prompt = prompts.TUTOR_PROMPT.format(context=self.format_context(hits), question=question)
        else:
            prompt = prompts.CLOSED_BOOK_PROMPT.format(question=question)

        system = prompts.tutor_system(level, language, grounded=use_retrieval,
                                      level_control=level_control, abstain=abstain)
        return hits, self.chat.stream(prompt, system, history=history, temperature=temperature)

    def make_quiz(self, topic: str, n: int = 5, level: str = "A2",
                  categories: list[str] | None = None) -> tuple[list[dict], list[SearchHit]]:
        if topic.strip():
            hits = self.retrieve(topic, k=6, categories=categories)
        else:
            hits = self.random_hits(6, categories)
            topic = "the content of these passages"
        if not hits:
            raise ValueError("Nothing in the knowledge base matches this topic. Try another topic.")

        raw = self.chat.generate(
            prompts.QUIZ_PROMPT.format(context=self.format_context(hits), n=n, topic=topic),
            prompts.QUIZ_SYSTEM.format(level=level), temperature=0.7, json_schema=prompts.QUIZ_SCHEMA)
        questions = json.loads(raw)["questions"]
        # Drop malformed questions instead of crashing the quiz
        valid = [q for q in questions if len(q.get("options", [])) >= 2
                 and 0 <= q.get("answer_index", -1) < len(q["options"])]
        return valid[:n], hits

    def explain_sentence(self, sentence: str, level: str = "A2") -> tuple[dict, list[SearchHit]]:
        hits = self.retrieve(f"German grammar: {sentence}", k=4, categories=["grammar"])
        raw = self.chat.generate(
            prompts.TRANSLATE_PROMPT.format(context=self.format_context(hits), sentence=sentence),
            prompts.TRANSLATE_SYSTEM.format(level=level), temperature=0.2,
            json_schema=prompts.TRANSLATE_SCHEMA)
        return json.loads(raw), hits

    def random_hits(self, k: int, categories: list[str] | None = None) -> list[SearchHit]:
        """Random chunks, for a "surprise me" quiz with no topic."""
        where = None
        if categories:
            where = {"category": categories[0]} if len(categories) == 1 else {"category": {"$in": categories}}
        data = self.store.collection.get(where=where, include=["documents", "metadatas"])
        pairs = list(zip(data["documents"], data["metadatas"]))
        random.shuffle(pairs)
        return [SearchHit(text=t, source=m["source"], category=m["category"], section=m.get("section", ""),
                          page=m.get("page"), score=1.0) for t, m in pairs[:k]]
