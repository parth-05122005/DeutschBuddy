"""Central settings, read from environment variables (or a .env file)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    # Chunking: characters per chunk and overlap between neighbouring chunks
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval: how many chunks to send to the LLM, and the minimum cosine similarity
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.45"))

    knowledge_dir: Path = PROJECT_ROOT / "knowledge_base"
    db_dir: Path = PROJECT_ROOT / "chroma_db"
    collection: str = "deutschbuddy"


settings = Settings()
