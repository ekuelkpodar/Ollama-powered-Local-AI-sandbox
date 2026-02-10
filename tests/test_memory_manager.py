import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from agent.config import AgentConfig, MemoryConfig
from memory.memory_manager import MemoryManager


class StubEmbeddingEngine:
    def __init__(self, dimension: int = 4):
        self.dimension = dimension

    def embed_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        score = sum(ord(c) for c in text) or 1
        vec[0] = float(score % 97)
        vec[1] = float((score * 3) % 101)
        vec[2] = float((score * 7) % 103)
        vec[3] = float((score * 11) % 107)
        norm = np.linalg.norm(vec)
        if norm == 0:
            norm = 1
        return vec / norm

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_single(t) for t in texts], dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):
    def _make_manager(self, memory_config: MemoryConfig, tmpdir: str) -> MemoryManager:
        config = AgentConfig(
            memory=memory_config,
            memory_dir=str(Path(tmpdir) / "memory"),
            log_dir=str(Path(tmpdir) / "logs"),
        )
        manager = MemoryManager(config)
        stub = StubEmbeddingEngine()
        manager.embedding_engine = stub
        manager._dimension = stub.get_dimension()
        return manager

    async def test_namespace_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(
                MemoryConfig(namespaces=["global", "project"]),
                tmpdir,
            )
            await manager.save("hello world", namespace="global")
            await manager.save("project note", namespace="project")

            global_results = await manager.search("hello", namespace="global")
            project_results = await manager.search("project", namespace="project")

            self.assertEqual(len(global_results), 1)
            self.assertEqual(len(project_results), 1)
            self.assertEqual(global_results[0]["content"], "hello world")

    async def test_ttl_prunes_expired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(
                MemoryConfig(ttl_days=1, namespaces=["default"]),
                tmpdir,
            )
            old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            await manager.save(
                "old memory",
                metadata={
                    "created_at": old_ts,
                    "timestamp": old_ts,
                    "last_accessed": old_ts,
                },
                namespace="default",
            )

            results = await manager.search("old", namespace="default")
            self.assertEqual(results, [])
            store = manager._get_store("default")
            self.assertEqual(store.count, 0)

    async def test_importance_scoring_orders_by_recency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(
                MemoryConfig(importance_scoring=True, namespaces=["default"]),
                tmpdir,
            )
            old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
            await manager.save(
                "older memory",
                metadata={"created_at": old_ts, "timestamp": old_ts},
                namespace="default",
            )
            await manager.save("newer memory", namespace="default")

            results = await manager.search("memory", namespace="default")
            self.assertEqual(results[0]["content"], "newer memory")

    async def test_compaction_creates_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(
                MemoryConfig(
                    compaction_enabled=True,
                    compaction_threshold=2,
                    namespaces=["default"],
                ),
                tmpdir,
            )
            await manager.save("first memory", namespace="default")
            await manager.save("second memory", namespace="default")
            await manager.save("third memory", namespace="default")

            store = manager._get_store("default")
            self.assertLessEqual(store.count, 3)
            self.assertTrue(any(m.get("compacted") for m in store.metadata))

    async def test_hybrid_search_adds_bm25_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_config = MemoryConfig(namespaces=["default"])
            manager = self._make_manager(memory_config, tmpdir)
            manager.config.rag.hybrid_search = True
            manager.config.rag.rerank_enabled = True

            await manager.save("alpha beta gamma", namespace="default")
            await manager.save("delta epsilon", namespace="default")

            results = await manager.search("alpha", namespace="default")
            self.assertTrue(results)
            self.assertIn("bm25_score", results[0])
            self.assertIn("hybrid_score", results[0])
