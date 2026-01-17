from __future__ import annotations

import asyncio
import json
import threading
from typing import Sequence

import aiohttp
import numpy as np

from config import EmbedderConfig, load_embedder_config


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, name="embedding-client")
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


class EmbeddingClient:
    def __init__(self, *, config: EmbedderConfig | None = None, model: str | None = None):
        self._config = config or load_embedder_config()
        self.model = model or self._config.model
        self._embeddings_url = self._config.embeddings_url
        self._headers = self._config.resolved_headers()

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return _run_coroutine_sync(self.embed_many_async(texts))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    async def embed_many_async(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        payload = {"input": list(texts), "model": self.model}
        response = await self._post_json(payload)

        if response.get("error"):
            raise RuntimeError(f"Embedding error: {response['error']}")

        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("Embedding response missing data")

        vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
        return self._normalize(vectors)

    async def _post_json(self, payload: dict) -> dict:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        headers.update(self._headers)

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._embeddings_url, json=payload, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    if resp.status == 401 and "Authorization" not in headers:
                        raise RuntimeError(
                            "Embedding request failed (401): missing API key. "
                            "Set GRAPHQL_EMBED_API_KEY or GRAPHQL_EMBED_HEADERS with Authorization."
                        )
                    raise RuntimeError(
                        f"Embedding request failed ({resp.status}): {text.strip()}"
                    )
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Embedding response was not valid JSON") from exc

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
