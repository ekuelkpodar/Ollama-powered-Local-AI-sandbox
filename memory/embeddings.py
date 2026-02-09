"""Local embedding engine using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Generates embeddings locally using sentence-transformers. No API needed."""

    _instances: dict[str, "EmbeddingEngine"] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingEngine":
        """Get or create a cached embedding engine instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dimension) numpy array, L2-normalized."""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dimension,) numpy array."""
        return self.model.encode([text], normalize_embeddings=True)[0]
