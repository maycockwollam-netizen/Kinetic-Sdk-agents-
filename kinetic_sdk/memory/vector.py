"""Optional embedding-backed, in-memory vector memory."""
from __future__ import annotations

import math
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any

from kinetic_sdk.memory.provider import (
    MemoryEntry,
    MemoryError,
    MemoryProvider,
    MemorySource,
    MemoryTier,
    expiry_iso,
    is_expired,
    matches_scope,
    utcnow_iso,
)
from kinetic_sdk.secret import SecretValue


class EmbeddingClient(ABC):
    """Provider-neutral embedding API; implementations return one vector/text."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LiteLLMEmbeddingClient(EmbeddingClient):
    """Lazy LiteLLM embedding wrapper (requires ``kinetic-agent-sdk[llm]``)."""

    def __init__(self, model: str, api_key: str | SecretValue | None = None) -> None:
        self.model, self._api_key = model, SecretValue(api_key) if isinstance(api_key, str) else api_key
        try:
            import litellm
        except ImportError as exc:
            raise ImportError("LiteLLMEmbeddingClient requires kinetic-agent-sdk[llm]") from exc
        self._litellm = litellm

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key.reveal()
        return [item["embedding"] for item in self._litellm.embedding(**kwargs).data]


class VectorMemory(MemoryProvider):
    """Cosine-ranked memory with the same tier, scope, and TTL contract."""

    def __init__(self, embeddings: EmbeddingClient) -> None:
        self._embeddings = embeddings
        self._entries: list[MemoryEntry] = []
        self._vectors: list[list[float] | None] = []
        self._lock = threading.Lock()

    def add(self, text: str, metadata: dict[str, Any] | None = None, *, tier: MemoryTier = MemoryTier.PERSISTENT, source: MemorySource = MemorySource.UNKNOWN, ttl_seconds: float | None = None) -> MemoryEntry:
        if not isinstance(text, str) or not text.strip():
            raise MemoryError("memory text must be a non-empty string")
        if not isinstance(tier, MemoryTier) or not isinstance(source, MemorySource):
            raise MemoryError("tier and source must be MemoryTier and MemorySource values")
        created_at = utcnow_iso()
        entry = MemoryEntry(uuid.uuid4().hex, text, dict(metadata or {}), created_at, tier, source, expiry_iso(ttl_seconds, created_at))
        try:
            vector = self._one(text)
        except Exception:
            vector = None
        with self._lock:
            self._entries.append(entry)
            self._vectors.append(vector)
        return entry

    def search(self, query: str, limit: int = 5, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        try:
            query_vector = self._one(query)
        except Exception:
            return []
        with self._lock:
            pairs = list(zip(self._entries, self._vectors))
        scored = [(self._cosine(query_vector, vector), index, entry) for index, (entry, vector) in enumerate(pairs) if vector is not None and (tiers is None or entry.tier in tiers) and matches_scope(entry, scope) and (include_expired or not is_expired(entry))]
        scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return [entry for score, _index, entry in scored if score > 0][:limit]

    def all(self, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        with self._lock:
            return [entry for entry in self._entries if (tiers is None or entry.tier in tiers) and matches_scope(entry, scope) and (include_expired or not is_expired(entry))]

    def clear(self, scope: dict[str, str] | None = None, *, tiers: set[MemoryTier] | None = None) -> None:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if not ((tiers is None or entry.tier in tiers) and matches_scope(entry, scope))]
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if entry.id != entry_id]
            deleted = len(keep) != len(self._entries)
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]
            return deleted

    def purge_expired(self) -> int:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if not is_expired(entry)]
            purged = len(self._entries) - len(keep)
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]
            return purged

    def _one(self, text: str) -> list[float]:
        vectors = self._embeddings.embed([text])
        if len(vectors) != 1 or not vectors[0]:
            raise MemoryError("embedding backend returned malformed vector")
        return [float(value) for value in vectors[0]]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0
