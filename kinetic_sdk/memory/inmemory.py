"""In-process, keyword-ranked memory with lifecycle filtering."""
from __future__ import annotations

import threading
import uuid
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
    relevance,
    utcnow_iso,
)


class InMemoryMemory(MemoryProvider):
    """Keyword-ranked process-local store; expired values are hidden by default."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._lock = threading.Lock()

    def add(self, text: str, metadata: dict[str, Any] | None = None, *, tier: MemoryTier = MemoryTier.PERSISTENT, source: MemorySource = MemorySource.UNKNOWN, ttl_seconds: float | None = None) -> MemoryEntry:
        if not isinstance(text, str) or not text.strip():
            raise MemoryError("memory text must be a non-empty string")
        if not isinstance(tier, MemoryTier) or not isinstance(source, MemorySource):
            raise MemoryError("tier and source must be MemoryTier and MemorySource values")
        created_at = utcnow_iso()
        entry = MemoryEntry(uuid.uuid4().hex, text, dict(metadata or {}), created_at, tier, source, expiry_iso(ttl_seconds, created_at))
        with self._lock:
            self._entries.append(entry)
        return entry

    def search(self, query: str, limit: int = 5, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        entries = self.all(tiers=tiers, scope=scope, include_expired=include_expired)
        scored = [(relevance(query, entry.text), index, entry) for index, entry in enumerate(entries)]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for score, _index, entry in scored if score > 0][:limit]

    def all(self, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        with self._lock:
            return [entry for entry in self._entries if (tiers is None or entry.tier in tiers) and matches_scope(entry, scope) and (include_expired or not is_expired(entry))]

    def clear(self, scope: dict[str, str] | None = None, *, tiers: set[MemoryTier] | None = None) -> None:
        with self._lock:
            self._entries = [entry for entry in self._entries if not ((tiers is None or entry.tier in tiers) and matches_scope(entry, scope))]

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            previous = len(self._entries)
            self._entries = [entry for entry in self._entries if entry.id != entry_id]
            return len(self._entries) != previous

    def purge_expired(self) -> int:
        with self._lock:
            previous = len(self._entries)
            self._entries = [entry for entry in self._entries if not is_expired(entry)]
            return previous - len(self._entries)
