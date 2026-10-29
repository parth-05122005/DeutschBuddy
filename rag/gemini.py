"""Thin wrappers around the Google Gemini API: embeddings and text generation."""

import re
import time
from typing import Iterator

import numpy as np
from google import genai
from google.genai import errors, types


def _retry_delay(exc: errors.APIError, attempt: int) -> float:
    """Seconds to wait: the delay the API asks for on a 429 (e.g. "retry in 51s"), else backoff."""
    match = re.search(r"retry in ([\d.]+)s|retryDelay': '(\d+)s", str(exc))
    if match:
        return min(float(match.group(1) or match.group(2)) + 2, 120)
    return 2 ** attempt * 2


def _with_retry(fn, attempts: int = 6):
    """Retry on rate limits (HTTP 429) and temporary server errors."""
    for attempt in range(attempts):
        try:
            return fn()
        except errors.APIError as exc:
            retryable = exc.code in (429, 500, 503)
            # A daily quota will not recover by waiting a minute
            if not retryable or attempt == attempts - 1 or "PerDay" in str(exc):
                raise
            time.sleep(_retry_delay(exc, attempt))


class GeminiEmbedder:
    """Multilingual embeddings: an English question can match German text."""

    batch_size = 50

    def __init__(self, client: genai.Client, model: str, dim: int):
        self.client = client
        self.model = model
        self.dim = dim
        self._query_cache: dict[str, list[float]] = {}

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            result = _with_retry(lambda: self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            ))
            vectors.extend(e.values for e in result.embeddings)
        # Reduced-dimension Gemini embeddings are not normalised, so normalise for cosine search
        arr = np.asarray(vectors, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        # The same question is often searched more than once; embedding it again wastes quota
        if text not in self._query_cache:
            self._query_cache[text] = self._embed([text], "RETRIEVAL_QUERY")[0]
        return self._query_cache[text]


class GeminiChat:
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    @staticmethod
    def _contents(prompt: str, history: list[dict] | None) -> list[types.Content]:
        contents = []
        for turn in history or []:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents

    def generate(self, prompt: str, system: str, temperature: float = 0.3,
                 json_schema: dict | None = None) -> str:
        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        if json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = json_schema
        response = _with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=self._contents(prompt, None), config=config))
        return response.text or ""

    def stream(self, prompt: str, system: str, history: list[dict] | None = None,
               temperature: float = 0.3) -> Iterator[str]:
        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)

        def start():
            # Rate-limit errors surface on the first chunk, so retry until it arrives
            parts = iter(self.client.models.generate_content_stream(
                model=self.model, contents=self._contents(prompt, history), config=config))
            return next(parts, None), parts

        first, rest = _with_retry(start)
        for part in ([first] if first is not None else []):
            if part.text:
                yield part.text
        for part in rest:
            if part.text:
                yield part.text
