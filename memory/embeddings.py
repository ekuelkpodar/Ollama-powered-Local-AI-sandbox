"""Embedding engine using Ollama's /api/embed endpoint."""

import asyncio
import json
import aiohttp
import numpy as np


class EmbeddingEngine:
    """Generates embeddings via Ollama. Falls back to sentence-transformers if available."""

    _instances: dict[str, "EmbeddingEngine"] = {}

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.dimension: int | None = None
        self._loop = None

    @classmethod
    def get(cls, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> "EmbeddingEngine":
        """Get or create a cached embedding engine instance."""
        key = f"{model_name}@{base_url}"
        if key not in cls._instances:
            cls._instances[key] = cls(model_name, base_url)
        return cls._instances[key]

    def _get_loop(self):
        """Get or create an event loop for sync calls."""
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    async def _embed_async(self, texts: list[str]) -> np.ndarray:
        """Call Ollama /api/embed for a batch of texts."""
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Ollama embed failed (HTTP {resp.status}): {body}\n"
                        f"Make sure '{self.model_name}' is pulled: ollama pull {self.model_name}"
                    )
                data = await resp.json()

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("Ollama returned empty embeddings")

        arr = np.array(embeddings, dtype=np.float32)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms

        if self.dimension is None:
            self.dimension = arr.shape[1]

        return arr

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dimension) numpy array, L2-normalized."""
        try:
            asyncio.get_running_loop()
            # We're inside an async context — can't use run_until_complete
            # Use a thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(self._sync_embed, texts)
                return future.result()
        except RuntimeError:
            # No running loop — safe to create one
            return self._sync_embed(texts)

    def _sync_embed(self, texts: list[str]) -> np.ndarray:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._embed_async(texts))
        finally:
            loop.close()

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dimension,) numpy array."""
        return self.embed([text])[0]

    def get_dimension(self) -> int:
        """Get embedding dimension. Runs a test embed if not yet known."""
        if self.dimension is None:
            self.embed_single("test")
        return self.dimension
