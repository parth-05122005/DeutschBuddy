"""Persistent vector database (ChromaDB) holding the embedded chunks."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .loader import Chunk


@dataclass
class SearchHit:
    text: str
    source: str
    category: str
    section: str
    page: int | None
    score: float  # cosine similarity, 1.0 = identical


class VectorStore:
    def __init__(self, path: Path, collection: str):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        return set(self.collection.get(ids=ids, include=[])["ids"])

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def search(self, embedding: list[float], k: int, categories: list[str] | None = None) -> list[SearchHit]:
        if self.count() == 0:
            return []
        where = None
        if categories:
            where = {"category": categories[0]} if len(categories) == 1 else {"category": {"$in": categories}}
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(k, self.count()), where=where,
            include=["documents", "metadatas", "distances"])
        hits = []
        for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append(SearchHit(
                text=text, source=meta.get("source", "?"), category=meta.get("category", ""),
                section=meta.get("section", ""), page=meta.get("page"), score=1.0 - dist))
        return hits

    def sources(self) -> dict[str, dict]:
        """All indexed files with their category and number of chunks."""
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        summary: dict[str, dict] = {}
        for meta in metas:
            entry = summary.setdefault(meta["source"], {"category": meta["category"], "chunks": 0})
            entry["chunks"] += 1
        return dict(sorted(summary.items()))

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
