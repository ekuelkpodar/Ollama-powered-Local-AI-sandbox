"""High-level memory API built on FAISS + embeddings."""

import math
import os
import time
import uuid
from datetime import datetime, timezone

from agent.config import AgentConfig
from agent.telemetry import Telemetry
from memory.embeddings import EmbeddingEngine, LocalEmbeddingEngine
from memory.bm25 import BM25Index
from memory.faiss_store import FAISSStore


# Memory area constants
AREA_MAIN = "main"
AREA_FRAGMENTS = "fragments"
AREA_SOLUTIONS = "solutions"
AREA_KNOWLEDGE = "knowledge"


class MemoryManager:
    """High-level memory operations: search, save, delete, forget."""

    def __init__(self, config: AgentConfig, telemetry: Telemetry | None = None):
        self.config = config
        self.telemetry = telemetry
        self.embedding_engine = self._select_embedding_engine()
        self._stores: dict[str, FAISSStore] = {}
        self._dimension: int | None = None
        self._last_prune: dict[str, datetime] = {}
        self._bm25_indexes: dict[str, BM25Index] = {}
        self._bm25_counts: dict[str, int] = {}
        self._bm25_dirty: dict[str, bool] = {}

    def _get_dimension(self) -> int:
        """Get embedding dimension, probing Ollama if needed."""
        if self._dimension is None:
            self._dimension = self.embedding_engine.get_dimension()
        return self._dimension

    def _get_store(self, namespace: str) -> FAISSStore:
        """Get or create a FAISS store for a namespace."""
        if namespace not in self._stores:
            directory = os.path.join(self.config.memory_dir, namespace)
            self._stores[namespace] = FAISSStore(directory, self._get_dimension())
        return self._stores[namespace]

    async def search(
        self,
        query: str,
        area: str | None = None,
        k: int = 5,
        threshold: float = 0.6,
        namespace: str | None = None,
        subdir: str = "default",
    ) -> list[dict]:
        """Semantic search over memories."""
        namespace = self._resolve_namespace(namespace, subdir)
        start_time = time.monotonic()

        def _finalize(results_list: list[dict]) -> list[dict]:
            duration_ms = (time.monotonic() - start_time) * 1000
            top_score = None
            if results_list:
                top_score = results_list[0].get("hybrid_score", results_list[0].get("score"))
            self._record_memory_operation(
                operation="search",
                query=query,
                result_count=len(results_list),
                top_score=top_score,
                namespace=namespace,
                duration_ms=duration_ms,
            )
            return results_list

        store = self._get_store(namespace)
        self._prune_expired(namespace, store)
        vector = self.embedding_engine.embed_single(query)
        search_k = max(k, 1) * 5
        vector_results = store.search(
            vector,
            k=search_k,
            area=area,
            threshold=threshold,
            include_indices=True,
        )
        bm25_results = []
        bm25_index = self._get_bm25_index(namespace, store)
        if bm25_index:
            bm25_results = bm25_index.search(query, top_k=search_k)

        if not vector_results and not bm25_results:
            return _finalize([])

        vector_map = {r.get("_index"): r.get("score", 0.0) for r in vector_results}
        bm25_map = {idx: score for idx, score in bm25_results}

        if self.config.rag.hybrid_search:
            candidate_indices = set(vector_map.keys()) | set(bm25_map.keys())
        else:
            candidate_indices = set(vector_map.keys())

        results: list[dict] = []
        for idx in candidate_indices:
            if idx is None or idx < 0 or idx >= len(store.metadata):
                continue
            meta = store.metadata[idx]
            if area and meta.get("area") != area:
                continue

            vector_score = float(vector_map.get(idx, 0.0))
            bm25_score = float(bm25_map.get(idx, 0.0))
            if vector_score and vector_score < threshold and not (
                self.config.rag.hybrid_search and bm25_score > 0
            ):
                continue

            result = {**meta, "score": vector_score}
            if bm25_score:
                result["bm25_score"] = bm25_score
            result["_index"] = int(idx)
            results.append(result)

        if not results:
            return _finalize([])

        now = datetime.now(timezone.utc)
        if self.config.rag.hybrid_search or self.config.rag.rerank_enabled:
            vector_weight, bm25_weight = self._normalize_weights(
                self.config.rag.vector_weight,
                self.config.rag.bm25_weight,
            )
            vector_max = max((r.get("score", 0.0) for r in results), default=0.0) or 1.0
            bm25_max = max((r.get("bm25_score", 0.0) for r in results), default=0.0) or 1.0
            for r in results:
                v = r.get("score", 0.0) / vector_max
                b = r.get("bm25_score", 0.0) / bm25_max
                r["hybrid_score"] = v * vector_weight + b * bm25_weight
            results.sort(key=lambda r: r.get("hybrid_score", r.get("score", 0)), reverse=True)

        indices = [r.pop("_index") for r in results if "_index" in r]
        self._update_access_metadata(store, indices, now)

        if self.config.memory.importance_scoring:
            for r in results:
                r["importance_score"] = self._importance_score(r, now)
            results.sort(key=lambda r: r.get("importance_score", r.get("score", 0)), reverse=True)

        return _finalize(results[:k])

    async def save(
        self,
        content: str,
        area: str = AREA_MAIN,
        metadata: dict | None = None,
        namespace: str | None = None,
        subdir: str = "default",
        allow_compaction: bool = True,
    ):
        """Save a new memory."""
        namespace = self._resolve_namespace(namespace, subdir)
        start_time = time.monotonic()
        vector = self.embedding_engine.embed_single(content)
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            "content": content,
            "area": area,
            "timestamp": now,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "namespace": namespace,
            "memory_id": uuid.uuid4().hex,
            **(metadata or {}),
        }
        store = self._get_store(namespace)
        store.add(vector.reshape(1, -1), [meta])
        self._bm25_dirty[namespace] = True
        if allow_compaction:
            self._maybe_compact(namespace, store)
        duration_ms = (time.monotonic() - start_time) * 1000
        self._record_memory_operation(
            operation="save",
            query=None,
            result_count=1,
            top_score=None,
            namespace=namespace,
            duration_ms=duration_ms,
        )

    async def delete(
        self,
        query: str,
        area: str | None = None,
        threshold: float = 0.8,
        namespace: str | None = None,
        subdir: str = "default",
    ):
        """Delete memories semantically similar to the query."""
        namespace = self._resolve_namespace(namespace, subdir)
        start_time = time.monotonic()
        results = await self.search(
            query,
            area=area,
            k=10,
            threshold=threshold,
            namespace=namespace,
            subdir=subdir,
        )
        if not results:
            duration_ms = (time.monotonic() - start_time) * 1000
            self._record_memory_operation(
                operation="delete",
                query=query,
                result_count=0,
                top_score=None,
                namespace=namespace,
                duration_ms=duration_ms,
            )
            return 0

        ids_to_delete = {r.get("memory_id") for r in results if r.get("memory_id")}
        contents_to_delete = {r["content"] for r in results if r.get("content")}
        store = self._get_store(namespace)
        store.delete(
            lambda m: (
                (m.get("memory_id") in ids_to_delete if ids_to_delete else False)
                or (m.get("content") in contents_to_delete)
            )
        )
        self._bm25_dirty[namespace] = True
        deleted_count = len(ids_to_delete) or len(contents_to_delete)
        duration_ms = (time.monotonic() - start_time) * 1000
        self._record_memory_operation(
            operation="delete",
            query=query,
            result_count=deleted_count,
            top_score=None,
            namespace=namespace,
            duration_ms=duration_ms,
        )
        return deleted_count

    async def forget(
        self,
        area: str,
        namespace: str | None = None,
        subdir: str = "default",
    ):
        """Clear all memories in a given area."""
        namespace = self._resolve_namespace(namespace, subdir)
        store = self._get_store(namespace)
        start_time = time.monotonic()
        before = store.count
        store.delete(lambda m: m.get("area") == area)
        self._bm25_dirty[namespace] = True
        after = store.count
        duration_ms = (time.monotonic() - start_time) * 1000
        self._record_memory_operation(
            operation="forget",
            query=None,
            result_count=max(before - after, 0),
            top_score=None,
            namespace=namespace,
            duration_ms=duration_ms,
        )

    async def get_stats(self, namespace: str | None = None, subdir: str = "default") -> dict:
        """Get memory statistics."""
        if namespace is None:
            namespaces = self._list_namespaces()
            total = 0
            by_namespace: dict[str, dict] = {}
            for ns in namespaces:
                store = self._get_store(ns)
                areas = self._count_areas(store)
                by_namespace[ns] = {
                    "total": store.count,
                    "areas": areas,
                }
                total += store.count
            return {"total": total, "namespaces": by_namespace}

        namespace = self._resolve_namespace(namespace, subdir)
        store = self._get_store(namespace)
        areas = self._count_areas(store)
        return {"total": store.count, "areas": areas, "namespace": namespace}

    async def list_memories(
        self,
        namespace: str | None = None,
        area: str | None = None,
        limit: int = 100,
        subdir: str = "default",
    ) -> list[dict]:
        """List recent memories by timestamp."""
        namespace = self._resolve_namespace(namespace, subdir)
        store = self._get_store(namespace)
        items: list[dict] = []
        for meta in store.metadata:
            if area and meta.get("area") != area:
                continue
            items.append({**meta})

        items.sort(
            key=lambda m: self._parse_timestamp(
                m.get("created_at") or m.get("timestamp") or m.get("last_accessed")
            ) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return items[: max(limit, 0)]

    async def delete_by_id(
        self,
        memory_id: str,
        namespace: str | None = None,
        subdir: str = "default",
    ) -> bool:
        """Delete a memory by ID."""
        namespace = self._resolve_namespace(namespace, subdir)
        store = self._get_store(namespace)
        start_time = time.monotonic()
        before = store.count
        store.delete(lambda m: m.get("memory_id") == memory_id)
        self._bm25_dirty[namespace] = True
        deleted = store.count < before
        duration_ms = (time.monotonic() - start_time) * 1000
        self._record_memory_operation(
            operation="delete_id",
            query=memory_id,
            result_count=1 if deleted else 0,
            top_score=None,
            namespace=namespace,
            duration_ms=duration_ms,
        )
        return deleted

    def _select_embedding_engine(self):
        provider = self.config.memory.embedding_provider
        if provider == "local":
            return LocalEmbeddingEngine.get(self.config.embedding_model)
        return EmbeddingEngine.get(
            self.config.memory.ollama_embedding_model,
            self.config.chat_model.base_url,
        )

    def _resolve_namespace(self, namespace: str | None, subdir: str) -> str:
        if namespace:
            return namespace
        if subdir and subdir != "default":
            return subdir
        namespaces = self._list_namespaces()
        return namespaces[0] if namespaces else "default"

    def _list_namespaces(self) -> list[str]:
        namespaces = self.config.memory.namespaces
        return namespaces if namespaces else ["default"]

    def _count_areas(self, store: FAISSStore) -> dict:
        areas: dict[str, int] = {}
        for m in store.metadata:
            a = m.get("area", "unknown")
            areas[a] = areas.get(a, 0) + 1
        return areas

    def _prune_expired(self, namespace: str, store: FAISSStore) -> None:
        ttl_days = self.config.memory.ttl_days
        if not ttl_days:
            return
        now = datetime.now(timezone.utc)
        last_prune = self._last_prune.get(namespace)
        if last_prune and (now - last_prune).total_seconds() < 60:
            return

        expired_ids: set[str] = set()
        expired_signatures: set[tuple[str, str]] = set()
        for meta in store.metadata:
            ts = self._parse_timestamp(meta.get("last_accessed") or meta.get("created_at") or meta.get("timestamp"))
            if ts is None:
                continue
            age_days = (now - ts).total_seconds() / 86400
            if age_days > ttl_days:
                mem_id = meta.get("memory_id")
                if mem_id:
                    expired_ids.add(mem_id)
                else:
                    content = meta.get("content")
                    timestamp = meta.get("timestamp")
                    if content and timestamp:
                        expired_signatures.add((content, timestamp))

        if expired_ids or expired_signatures:
            store.delete(
                lambda m: (
                    m.get("memory_id") in expired_ids
                    or (
                        not m.get("memory_id")
                        and (m.get("content"), m.get("timestamp")) in expired_signatures
                    )
                )
            )
            self._bm25_dirty[namespace] = True
        self._last_prune[namespace] = now

    def _update_access_metadata(self, store: FAISSStore, indices: list[int], now: datetime) -> None:
        if not indices:
            return

        def update(meta: dict) -> None:
            meta["last_accessed"] = now.isoformat()
            meta["access_count"] = int(meta.get("access_count", 0)) + 1

        store.update_metadata_bulk(indices, update)

    def _importance_score(self, meta: dict, now: datetime) -> float:
        relevance = float(meta.get("hybrid_score", meta.get("score", 0.0)))
        access_count = int(meta.get("access_count", 0))
        freq_score = 1.0 + math.log1p(access_count)

        ts = self._parse_timestamp(meta.get("last_accessed") or meta.get("created_at") or meta.get("timestamp"))
        if ts is None:
            recency_score = 1.0
        else:
            age_days = max((now - ts).total_seconds() / 86400, 0.0)
            ttl_days = self.config.memory.ttl_days
            if ttl_days:
                recency_score = math.exp(-age_days / max(ttl_days, 1))
            else:
                recency_score = 1 / (1 + age_days)

        return relevance * freq_score * recency_score

    def _maybe_compact(self, namespace: str, store: FAISSStore) -> None:
        if not self.config.memory.compaction_enabled:
            return
        threshold = self.config.memory.compaction_threshold
        if store.count <= threshold:
            return

        candidates = []
        for idx, meta in enumerate(store.metadata):
            if meta.get("area") == AREA_KNOWLEDGE:
                continue
            if meta.get("compacted"):
                continue
            ts = self._parse_timestamp(meta.get("created_at") or meta.get("timestamp"))
            if ts is None:
                continue
            candidates.append((idx, ts, meta))

        if len(candidates) < 2:
            return

        candidates.sort(key=lambda item: item[1])
        batch_size = max(2, min(50, len(candidates), store.count - threshold + 1))
        selected = candidates[:batch_size]
        summary = self._summarize_memories([meta.get("content", "") for _, _, meta in selected])
        if not summary:
            return

        compacted_ids = {meta.get("memory_id") for _, _, meta in selected if meta.get("memory_id")}
        meta = {
            "compacted": True,
            "compacted_count": len(selected),
            "compacted_from": list(compacted_ids),
        }
        # Save summary without recursive compaction.
        vector = self.embedding_engine.embed_single(summary)
        now = datetime.now(timezone.utc).isoformat()
        summary_meta = {
            "content": summary,
            "area": "compacted",
            "timestamp": now,
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "namespace": namespace,
            "memory_id": uuid.uuid4().hex,
            **meta,
        }
        store.add(vector.reshape(1, -1), [summary_meta])
        if compacted_ids:
            store.delete(lambda m: m.get("memory_id") in compacted_ids)
        self._bm25_dirty[namespace] = True

    @staticmethod
    def _summarize_memories(contents: list[str]) -> str:
        lines = []
        for content in contents:
            content = content.strip()
            if not content:
                continue
            snippet = MemoryManager._first_sentence(content)
            if not snippet:
                snippet = content[:160]
            lines.append(f"- {snippet.strip()}")
        return "Compacted memory summary:\n" + "\n".join(lines) if lines else ""

    @staticmethod
    def _first_sentence(text: str) -> str:
        for sep in (".", "!", "?"):
            idx = text.find(sep)
            if idx != -1 and idx < 200:
                return text[: idx + 1]
        return text[:160]

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC")
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _get_bm25_index(self, namespace: str, store: FAISSStore) -> BM25Index | None:
        if not (self.config.rag.hybrid_search or self.config.rag.rerank_enabled):
            return None
        current_count = store.count
        if self._bm25_dirty.get(namespace) or self._bm25_counts.get(namespace) != current_count:
            documents = self._build_bm25_documents(store)
            self._bm25_indexes[namespace] = BM25Index(documents)
            self._bm25_counts[namespace] = current_count
            self._bm25_dirty[namespace] = False
        return self._bm25_indexes.get(namespace)

    def _record_memory_operation(
        self,
        operation: str,
        query: str | None,
        result_count: int,
        top_score: float | None,
        namespace: str | None,
        duration_ms: float,
    ) -> None:
        if self.telemetry is None:
            return
        self.telemetry.record_memory_operation(
            operation=operation,
            query=query,
            result_count=result_count,
            top_score=top_score,
            namespace=namespace,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _build_bm25_documents(store: FAISSStore) -> list[str]:
        docs: list[str] = []
        for meta in store.metadata:
            content = meta.get("content", "")
            title = meta.get("title", "")
            headers = meta.get("section_headers", [])
            header_text = " ".join(headers) if isinstance(headers, list) else str(headers)
            docs.append(" ".join([title, header_text, content]).strip())
        return docs

    @staticmethod
    def _normalize_weights(vector_weight: float, bm25_weight: float) -> tuple[float, float]:
        total = vector_weight + bm25_weight
        if total <= 0:
            return 1.0, 0.0
        return vector_weight / total, bm25_weight / total
