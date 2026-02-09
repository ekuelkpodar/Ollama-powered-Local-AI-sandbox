"""High-level memory API built on FAISS + sentence-transformers."""

import os
from datetime import datetime, timezone

from agent.config import AgentConfig
from memory.embeddings import EmbeddingEngine
from memory.faiss_store import FAISSStore


# Memory area constants
AREA_MAIN = "main"
AREA_FRAGMENTS = "fragments"
AREA_SOLUTIONS = "solutions"
AREA_KNOWLEDGE = "knowledge"


class MemoryManager:
    """High-level memory operations: search, save, delete, forget."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.embedding_engine = EmbeddingEngine.get(config.embedding_model)
        self._stores: dict[str, FAISSStore] = {}

    def _get_store(self, subdir: str = "default") -> FAISSStore:
        """Get or create a FAISS store for a subdirectory."""
        if subdir not in self._stores:
            directory = os.path.join(self.config.memory_dir, subdir)
            self._stores[subdir] = FAISSStore(
                directory, self.embedding_engine.dimension
            )
        return self._stores[subdir]

    async def search(
        self,
        query: str,
        area: str | None = None,
        k: int = 5,
        threshold: float = 0.6,
        subdir: str = "default",
    ) -> list[dict]:
        """Semantic search over memories."""
        vector = self.embedding_engine.embed_single(query)
        store = self._get_store(subdir)
        return store.search(vector, k=k, area=area, threshold=threshold)

    async def save(
        self,
        content: str,
        area: str = AREA_MAIN,
        metadata: dict | None = None,
        subdir: str = "default",
    ):
        """Save a new memory."""
        vector = self.embedding_engine.embed_single(content)
        meta = {
            "content": content,
            "area": area,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        store = self._get_store(subdir)
        store.add(vector.reshape(1, -1), [meta])

    async def delete(
        self,
        query: str,
        area: str | None = None,
        threshold: float = 0.8,
        subdir: str = "default",
    ):
        """Delete memories semantically similar to the query."""
        results = await self.search(query, area=area, k=10, threshold=threshold, subdir=subdir)
        if not results:
            return 0

        contents_to_delete = {r["content"] for r in results}
        store = self._get_store(subdir)
        store.delete(lambda m: m.get("content") in contents_to_delete)
        return len(contents_to_delete)

    async def forget(self, area: str, subdir: str = "default"):
        """Clear all memories in a given area."""
        store = self._get_store(subdir)
        store.delete(lambda m: m.get("area") == area)

    async def get_stats(self, subdir: str = "default") -> dict:
        """Get memory statistics."""
        store = self._get_store(subdir)
        areas = {}
        for m in store.metadata:
            a = m.get("area", "unknown")
            areas[a] = areas.get(a, 0) + 1
        return {"total": store.count, "areas": areas}
