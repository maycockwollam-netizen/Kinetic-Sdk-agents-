"""In-process memory provider (zero dependencies).

Stores entries in a list and ranks them by the keyword-overlap
:func:`~kinetic_sdk.memory.provider.relevance` score. Ties break towards
the NEWER entry (recall over time decays: an agent should prefer the
session from yesterday over the one from last month when both match).

Suitable for tests, demos and small deployments. Swap in
:class:`~kinetic_sdk.memory.json_file.JsonFileMemory` for persistence, or
your own embedding-backed provider implementing the same ABC for scale.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from kinetic_sdk.memory.provider import (
    MemoryEntry,
    MemoryError,
    MemoryProvider,
    relevance,
    utcnow_iso,
)


class InMemoryMemory(MemoryProvider):
    """Keyword-ranked, process-local memory store."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        # Threads share an agent's memory when sub-agents run in parallel;
        # a plain list append is GIL-safe but search+add must not interleave
        # half-built entries, so guard with a lock.
        self._lock = threading.Lock()

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Store *text*; blank text raises :class:`MemoryError`."""
        if not isinstance(text, str) or not text.strip():
            raise MemoryError("memory text must be a non-empty string")
        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            text=text,
            metadata=dict(metadata or {}),
            created_at=utcnow_iso(),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Rank by relevance; entries scoring 0 fall out (no noise recall)."""
        with self._lock:
            scored = [
                (relevance(query, e.text), i, e) for i, e in enumerate(self._entries)
            ]
        # Newer first on ties: index descending as the tie-breaker.
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [e for score, _i, e in scored if score > 0][:limit]

    def all(self) -> list[MemoryEntry]:
        """Every entry, oldest first."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
