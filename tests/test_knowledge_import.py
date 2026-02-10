import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.config import AgentConfig, RagConfig
from memory.knowledge_import import KnowledgeImporter


class StubEmbeddingEngine:
    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            score = float(len(text) or 1)
            vec = np.array([score, score / 2, score / 3], dtype=np.float32)
            norm = np.linalg.norm(vec)
            vecs.append(vec / (norm or 1.0))
        return np.stack(vecs, axis=0)


class FakeMemoryManager:
    def __init__(self, config):
        self.config = config
        self.embedding_engine = StubEmbeddingEngine()
        self.saved = []

    async def save(self, content: str, area: str, metadata: dict, namespace: str | None = None, subdir: str = "default"):
        self.saved.append((content, area, metadata, namespace, subdir))


class TestKnowledgeImporter(unittest.TestCase):
    def _make_importer(self, rag_config: RagConfig) -> KnowledgeImporter:
        config = AgentConfig(
            rag=rag_config,
            data_dir=tempfile.mkdtemp(),
            knowledge_dir=tempfile.mkdtemp(),
        )
        mm = FakeMemoryManager(config)
        return KnowledgeImporter(mm)

    def test_fixed_chunking(self):
        rag = RagConfig(chunk_strategy="fixed", chunk_size=5, chunk_overlap=0)
        importer = self._make_importer(rag)
        chunks = importer._chunk_text("abcdefghij")
        self.assertEqual(chunks, ["abcde", "fghij"])

    def test_sentence_chunking(self):
        rag = RagConfig(chunk_strategy="sentence", chunk_size=20, chunk_overlap=0)
        importer = self._make_importer(rag)
        text = "First sentence. Second sentence. Third sentence."
        chunks = importer._chunk_text(text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.endswith(".") for chunk in chunks))

    def test_metadata_extraction(self):
        rag = RagConfig()
        importer = self._make_importer(rag)
        text = "# Title\nDate: 2024-01-01\n\n## Section\nContent"
        meta = importer._extract_metadata(text, "/tmp/sample.md", ".md")
        self.assertEqual(meta["title"], "Title")
        self.assertEqual(meta["date"], "2024-01-01")
        self.assertIn("Section", meta["section_headers"])
