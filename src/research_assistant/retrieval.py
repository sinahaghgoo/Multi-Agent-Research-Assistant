"""Hybrid passage retrieval: BM25 (exact terms such as "Article 51" or "10^25") fused with
dense embeddings (paraphrases) via reciprocal rank fusion.

The local corpus has ~1,000 passages, so a NumPy matrix is sufficient; no vector database.
If the embedding model is unavailable the index degrades to BM25 and records a warning.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

from .schemas import Passage

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "what",
        "when",
        "how",
        "does",
        "do",
        "shall",
        "may",
        "can",
        "under",
        "any",
        "such",
        "their",
        "they",
        "these",
        "those",
    ]
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []

    def add(self, docs: Iterable[list[str]]) -> None:
        for tokens in docs:
            idx = len(self.doc_len)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))

    def scores(self, query: list[str]) -> np.ndarray:
        n = len(self.doc_len)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0:
            return scores
        avg_len = sum(self.doc_len) / n
        for term in set(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, tf in postings:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / avg_len)
                scores[idx] += idf * tf * (self.k1 + 1) / norm
        return scores


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Wraps langchain-ollama embeddings and applies the model's query/document prefixes."""

    QUERY_TASK = "Given a research question, retrieve document passages that help answer it"

    def __init__(self, model: str, base_url: str, batch_size: int = 32) -> None:
        from langchain_ollama import OllamaEmbeddings

        self.name = model
        self.batch_size = batch_size
        self._client = OllamaEmbeddings(model=model, base_url=base_url)

    def _query_prefix(self) -> str:
        if "qwen3-embedding" in self.name:
            return f"Instruct: {self.QUERY_TASK}\nQuery:"
        if "nomic-embed" in self.name:
            return "search_query: "
        return ""

    def _doc_prefix(self) -> str:
        return "search_document: " if "nomic-embed" in self.name else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        prefix = self._doc_prefix()
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._client.embed_documents([prefix + t for t in texts[i : i + self.batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(self._query_prefix() + text)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class PassageIndex:
    def __init__(
        self,
        passages: list[Passage],
        source_titles: dict[str, str],
        embedder: Embedder | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.passages: list[Passage] = []
        self.by_id: dict[str, int] = {}
        self.source_titles = dict(source_titles)
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.bm25 = BM25()
        self.vectors: np.ndarray | None = None
        self.warnings: list[str] = []
        self.add(passages, use_cache=True)

    @property
    def dense_available(self) -> bool:
        return self.embedder is not None and self.vectors is not None

    def _index_text(self, passage: Passage) -> str:
        title = self.source_titles.get(passage.source_id, passage.source_id)
        return f"{title} — {passage.location}\n{passage.text}"

    def _embed(self, texts: list[str], use_cache: bool) -> np.ndarray | None:
        if self.embedder is None:
            return None
        cache_file = None
        if use_cache and self.cache_dir is not None:
            digest = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16]
            slug = re.sub(r"[^a-z0-9]+", "-", self.embedder.name.lower())
            cache_file = self.cache_dir / "embeddings" / f"{slug}-{digest}.npy"
            if cache_file.exists():
                return np.load(cache_file)
        try:
            matrix = _normalise(np.asarray(self.embedder.embed_documents(texts), dtype=np.float32))
        except Exception as exc:  # embedding server down, model missing, timeout
            self.warnings.append(f"embedding failed ({type(exc).__name__}: {exc}); using BM25 only")
            self.embedder = None
            self.vectors = None
            return None
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, matrix)
        return matrix

    def add(self, passages: list[Passage], use_cache: bool = False) -> None:
        new = [p for p in passages if p.id not in self.by_id]
        if not new:
            return
        texts = [self._index_text(p) for p in new]
        for p in new:
            self.by_id[p.id] = len(self.passages)
            self.passages.append(p)
        self.bm25.add(tokenize(t) for t in texts)
        matrix = self._embed(texts, use_cache)
        if matrix is not None:
            self.vectors = matrix if self.vectors is None else np.vstack([self.vectors, matrix])

    def get(self, passage_id: str) -> Passage | None:
        idx = self.by_id.get(passage_id)
        return self.passages[idx] if idx is not None else None

    def search(
        self, query: str, k: int = 5, source_ids: set[str] | None = None, pool: int = 50
    ) -> list[tuple[Passage, float]]:
        allowed = np.ones(len(self.passages), dtype=bool)
        if source_ids:
            allowed = np.array([p.source_id in source_ids for p in self.passages])

        fused: dict[int, float] = defaultdict(float)
        bm25 = np.where(allowed, self.bm25.scores(tokenize(query)), -np.inf)
        for rank, idx in enumerate(np.argsort(-bm25)[:pool]):
            if np.isfinite(bm25[idx]) and bm25[idx] > 0:
                fused[int(idx)] += 1.0 / (RRF_K + rank + 1)

        if self.dense_available:
            try:
                q = _normalise(np.asarray([self.embedder.embed_query(query)], dtype=np.float32))[0]
                dense = np.where(allowed, self.vectors @ q, -np.inf)
                for rank, idx in enumerate(np.argsort(-dense)[:pool]):
                    fused[int(idx)] += 1.0 / (RRF_K + rank + 1)
            except Exception as exc:
                self.warnings.append(f"query embedding failed ({type(exc).__name__}); BM25 only")

        ranked = sorted(fused.items(), key=lambda item: -item[1])[:k]
        return [(self.passages[idx], score) for idx, score in ranked]
