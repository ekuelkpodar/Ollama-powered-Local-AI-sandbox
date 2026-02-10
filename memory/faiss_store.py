"""FAISS vector store wrapper with metadata and persistence."""

import os
import pickle
from typing import Callable

import faiss
import numpy as np


class FAISSStore:
    """Wraps a FAISS IndexFlatIP with parallel metadata storage and disk persistence."""

    def __init__(self, directory: str, dimension: int = 384):
        self.directory = directory
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._load_or_create()

    def _index_path(self) -> str:
        return os.path.join(self.directory, "index.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.directory, "metadata.pkl")

    def _load_or_create(self):
        """Load existing index from disk or create a new one."""
        os.makedirs(self.directory, exist_ok=True)
        if os.path.exists(self._index_path()) and os.path.exists(self._meta_path()):
            self.index = faiss.read_index(self._index_path())
            with open(self._meta_path(), "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        """
        Add vectors with metadata.
        vectors: shape (N, dimension), float32, L2-normalized
        metadata_list: list of dicts, one per vector
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)
        self._save()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        area: str | None = None,
        threshold: float = 0.0,
        include_indices: bool = False,
    ) -> list[dict]:
        """
        Search for similar vectors.
        Returns list of metadata dicts with added 'score' field.
        Filters by area and minimum similarity threshold.
        """
        if self.index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        search_k = min(k * 5, self.index.ntotal)
        scores, indices = self.index.search(query, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if area and meta.get("area") != area:
                continue
            result = {**meta, "score": float(score)}
            if include_indices:
                result["_index"] = int(idx)
            results.append(result)
            if len(results) >= k:
                break

        return results

    def delete(self, filter_fn: Callable[[dict], bool]):
        """
        Delete entries matching the filter function.
        Rebuilds the index (FAISS doesn't support native deletion on IndexFlatIP).
        """
        keep_indices = [i for i, m in enumerate(self.metadata) if not filter_fn(m)]
        if len(keep_indices) == len(self.metadata):
            return  # Nothing to delete

        if not keep_indices:
            # Delete everything
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self._save()
            return

        # Reconstruct vectors for kept entries
        vectors = np.array(
            [self.index.reconstruct(i) for i in keep_indices],
            dtype=np.float32,
        )
        new_metadata = [self.metadata[i] for i in keep_indices]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.metadata = new_metadata
        self._save()

    def clear(self):
        """Remove all entries."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save()

    def update_metadata_bulk(self, indices: list[int], update_fn: Callable[[dict], None]):
        """Update metadata in bulk and persist changes."""
        updated = False
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                update_fn(self.metadata[idx])
                updated = True
        if updated:
            self._save()

    @property
    def count(self) -> int:
        return self.index.ntotal

    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self.metadata, f)
