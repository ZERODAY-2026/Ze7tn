"""Local retrieval. Default: in-memory lexical (fully offline, no embeddings)."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

from src.chunking import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        data = self.chunk.as_dict()
        data["score"] = self.score
        return data


class MemoryIndex:
    """TF-IDF-like cosine over bag-of-words. No model download, no network."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self._docs = [_tokenize(c.text) for c in self.chunks]
        df: Counter[str] = Counter()
        for tokens in self._docs:
            df.update(set(tokens))
        n = max(len(self._docs), 1)
        self._idf = {term: math.log((1 + n) / (1 + df[term])) + 1.0 for term in df}
        self._vecs = [_tfidf(tokens, self._idf) for tokens in self._docs]

    def query(self, text: str, k: int = 6) -> list[RetrievedPassage]:
        if not self.chunks:
            return []
        q = _tfidf(_tokenize(text), self._idf)
        scored = [
            RetrievedPassage(chunk=chunk, score=_cosine(q, vec))
            for chunk, vec in zip(self.chunks, self._vecs)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        hits = [h for h in scored if h.score > 0][:k]
        if hits:
            return hits
        # If lexical scores are all zero, still return leading chunks so the
        # model can refuse from real text rather than hallucinate.
        return [
            RetrievedPassage(chunk=c, score=0.0) for c in self.chunks[: min(k, len(self.chunks))]
        ]


class LocalStore:
    """Facade used by the agent. Default backend is in-memory lexical."""

    def __init__(self, chunks: list[Chunk], backend: str | None = None):
        _offline_env()
        self.backend = (backend or os.environ.get("LEGAL_TRIAGE_RETRIEVER") or "memory").lower()
        self.chunks = chunks
        if self.backend == "chroma":
            self._chroma = _ChromaIndex(chunks)
            self._memory = None
        else:
            self._chroma = None
            self._memory = MemoryIndex(chunks)

    def query(self, text: str, k: int = 6) -> list[RetrievedPassage]:
        if self._chroma is not None:
            return self._chroma.query(text, k=k)
        assert self._memory is not None
        return self._memory.query(text, k=k)


class _ChromaIndex:
    """Optional local Chroma. Requires a pre-cached embedding model; no cloud."""

    def __init__(self, chunks: list[Chunk]):
        _offline_env()
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Use LEGAL_TRIAGE_RETRIEVER=memory."
            ) from exc

        settings = Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=False)
        client = chromadb.Client(settings)
        collection = client.create_collection(
            name="legal_triage_local",
            metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "page": c.page,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "document_name": c.document_name,
                    }
                    for c in chunks
                ],
            )
        self._by_id = {c.id: c for c in chunks}
        self._collection = collection

    def query(self, text: str, k: int = 6) -> list[RetrievedPassage]:
        if not self._by_id:
            return []
        n = min(k, len(self._by_id))
        result = self._collection.query(query_texts=[text], n_results=n)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[RetrievedPassage] = []
        for cid, dist in zip(ids, distances):
            chunk = self._by_id.get(cid)
            if chunk is None:
                continue
            score = 1.0 / (1.0 + float(dist))
            hits.append(RetrievedPassage(chunk=chunk, score=score))
        return hits


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    n = max(len(tokens), 1)
    return {term: (count / n) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
