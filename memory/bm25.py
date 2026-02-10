"""Lightweight BM25 implementation for keyword retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter


class BM25Index:
    """BM25 index over a list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.idf = self._build_idf(self.doc_tokens)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Return ranked (doc_index, score) results."""
        if not self.documents:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = self._score(tokens, doc_tokens, self.doc_len[idx])
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        if top_k is not None:
            return scores[:top_k]
        return scores

    def _score(self, query_tokens: list[str], doc_tokens: list[str], doc_len: int) -> float:
        tf = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = self.idf.get(token, 0.0)
            freq = tf[token]
            denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        return score

    @staticmethod
    def _build_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
        doc_count = len(tokenized_docs)
        df: Counter[str] = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] += 1
        idf = {}
        for token, freq in df.items():
            idf[token] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())
