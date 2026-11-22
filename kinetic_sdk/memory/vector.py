"""Embedding-backed vector memory, optionally persisted to one JSON file.

The in-memory cosine store remains the default (zero-dependency). Passing a
``path`` additionally persists entries AND their embedding vectors between
process runs via an atomic tmp-file + rename (same pattern as
``conversation/store.py``), so a restarted agent recalls the same vectors
without re-embedding everything.

Security note (same as the JSON keyword memory): the persisted file contains
RAW memory content. Protect it at the filesystem level; providers never
redact (redaction would corrupt recall).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from kinetic_sdk import _kinetic_rs  # type: ignore[attr-defined]  # extension built by maturin
except ImportError:  # Source checkout or Python-only environment without extension.
    _kinetic_rs = None

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
    """Cosine-ranked memory with the same tier, scope, and TTL contract.

    Args:
        embeddings: Backend producing one vector per text (``None`` vectors
            are stored but excluded from search, matching the fail-soft rule).
        path: Optional JSON file the entries AND their vectors are persisted
            to after every mutation and loaded at construction. ``None``
            keeps the historical purely in-memory behaviour.
    """

    #: On-disk layout version. Loaders reject anything newer.
    SCHEMA_VERSION = 1

    def __init__(self, embeddings: EmbeddingClient, path: str | os.PathLike[str] | None = None) -> None:
        self._embeddings = embeddings
        self._entries: list[MemoryEntry] = []
        self._vectors: list[list[float] | None] = []
        self._lock = threading.Lock()
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self._load()

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
        self._persist()
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
        filtered = [
            (index, entry, vector)
            for index, (entry, vector) in enumerate(pairs)
            if vector is not None
            and (tiers is None or entry.tier in tiers)
            and matches_scope(entry, scope)
            and (include_expired or not is_expired(entry))
        ]
        if not filtered:
            return []
        if _kinetic_rs is not None:
            try:
                ranked = _kinetic_rs.cosine_top_k(
                    query_vector,
                    [index for index, _entry, _vector in filtered],
                    [vector for _index, _entry, vector in filtered],
                    limit,
                )
                by_index = {index: entry for index, entry, _vector in filtered}
                return [by_index[index] for index, _score in ranked]
            except Exception:  # noqa: BLE001 - ranking must never break search
                pass  # fall back to the pure-Python ranking below
        scored = [(self._cosine(query_vector, vector), index, entry) for index, entry, vector in filtered]
        scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return [entry for score, _index, entry in scored if score > 0][:limit]

    def all(self, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        with self._lock:
            return [entry for entry in self._entries if (tiers is None or entry.tier in tiers) and matches_scope(entry, scope) and (include_expired or not is_expired(entry))]

    def clear(self, scope: dict[str, str] | None = None, *, tiers: set[MemoryTier] | None = None) -> None:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if not ((tiers is None or entry.tier in tiers) and matches_scope(entry, scope))]
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]
        self._persist()

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if entry.id != entry_id]
            deleted = len(keep) != len(self._entries)
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]
        if deleted:
            self._persist()
        return deleted

    def purge_expired(self) -> int:
        with self._lock:
            keep = [(entry, vector) for entry, vector in zip(self._entries, self._vectors) if not is_expired(entry)]
            purged = len(self._entries) - len(keep)
            self._entries, self._vectors = [entry for entry, _ in keep], [vector for _, vector in keep]
        if purged:
            self._persist()
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

    # --- persistence ---------------------------------------------------

    def _persist(self) -> None:
        """Atomically write entries + vectors to ``self.path`` (no-op in-memory)."""
        if self.path is None:
            return
        with self._lock:
            entries = [asdict(entry) for entry in self._entries]
            vectors = [list(vector) if vector is not None else None for vector in self._vectors]
        payload = {"schema_version": self.SCHEMA_VERSION, "entries": entries, "vectors": vectors}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        """Load a previously persisted store; broken files fail loudly."""
        if self.path is None or not self.path.exists():
            return
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryError(f"Cannot load vector memory from {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or "entries" not in payload or "vectors" not in payload:
            raise MemoryError(f"Cannot load vector memory from {self.path}: not a memory file")
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise MemoryError(
                f"Cannot load vector memory from {self.path}: schema version "
                f"{payload.get('schema_version')!r} is not supported (expected {self.SCHEMA_VERSION})"
            )
        raw_entries = payload["entries"]
        raw_vectors = payload["vectors"]
        if not isinstance(raw_entries, list) or not isinstance(raw_vectors, list):
            raise MemoryError(f"Cannot load vector memory from {self.path}: entries/vectors not a list")
        if len(raw_entries) != len(raw_vectors):
            raise MemoryError(
                f"Cannot load vector memory from {self.path}: entries/vectors length mismatch"
            )
        entries: list[MemoryEntry] = []
        vectors: list[list[float] | None] = []
        for item, vector in zip(raw_entries, raw_vectors):
            if not isinstance(item, dict):
                raise MemoryError(f"Cannot load vector memory from {self.path}: bad entry")
            entries.append(_entry_from_dict(item))
            vectors.append(_vector_from_any(vector))
        with self._lock:
            self._entries = entries
            self._vectors = vectors


def _vector_from_any(value: Any) -> list[float] | None:
    """Accept a JSON list of floats or ``None`` (malformed -> fails load)."""
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        raise MemoryError("vector memory file contains a malformed vector")
    return [float(item) for item in value]


def _entry_from_dict(data: dict[str, Any]) -> MemoryEntry:
    """Rebuild one MemoryEntry from its serialised dict (strict validation)."""
    text = data.get("text")
    entry_id = data.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise MemoryError("vector memory file contains an entry without an id")
    if not isinstance(text, str) or not text:
        raise MemoryError("vector memory file contains an entry with empty text")
    tier = data.get("tier")
    source = data.get("source")
    try:
        tier_value = MemoryTier(tier) if isinstance(tier, str) else MemoryTier.PERSISTENT
        source_value = MemorySource(source) if isinstance(source, str) else MemorySource.UNKNOWN
    except ValueError:
        raise MemoryError("vector memory file contains an entry with an unknown tier/source") from None
    expires_at = data.get("expires_at")
    return MemoryEntry(
        id=entry_id,
        text=text,
        metadata=dict(data.get("metadata") or {}),
        created_at=str(data.get("created_at", "")),
        tier=tier_value,
        source=source_value,
        expires_at=expires_at if isinstance(expires_at, str) else None,
    )
