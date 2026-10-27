"""Thin wrappers around the Google Gemini API: embeddings and text generation."""

import time
from typing import Iterator

import numpy as np
from google import genai
from google.genai import errors, types


def _with_retry(fn, attempts: int = 4):
    """Retry on rate limits (HTTP 429) and temporary server errors, with backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except errors.APIError as exc:
            retryable = exc.code in (429, 500, 503)
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 2)


class GeminiEmbedder:
    """Multilingual embeddings: an English question can match German text."""

    batch_size = 50

    def __init__(self, client: genai.Client, model: str, dim: int):
        self.client = client
        self.model = model
        self.dim = dim

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
        return self._embed([text], "RETRIEVAL_QUERY")[0]


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
        for part in self.client.models.generate_content_stream(
                model=self.model, contents=self._contents(prompt, history), config=config):
            if part.text:
                yield part.text
