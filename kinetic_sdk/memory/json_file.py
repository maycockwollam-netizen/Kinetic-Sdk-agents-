"""JSON-file-backed memory provider.

Same keyword-overlap search as
:class:`~kinetic_sdk.memory.inmemory.InMemoryMemory`, persisted to one JSON
file (atomic tmp-file + rename, mirroring ``conversation/store.py``).

Security note (same as the conversation store): the file contains RAW
memory content — user questions and assistant answers. Protect it at the
filesystem level; providers never redact (redaction would corrupt recall).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from kinetic_sdk.memory.inmemory import InMemoryMemory
from kinetic_sdk.memory.provider import (
    MemoryEntry,
    MemoryError,
    MemorySource,
    MemoryTier,
)

SCHEMA_VERSION = 2


class JsonFileMemory(InMemoryMemory):
    """Persistent memory: in-memory ranking, JSON file durability.

    Writes happen after every mutating call (memory is small — a few
    hundred entries — so full-file rewrites stay cheap). A corrupted or
    wrong-version file raises :class:`MemoryError` on construction rather
    than silently starting fresh, same contract as the conversation store.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._load()

    def add(self, text: str, metadata: dict[str, Any] | None = None, *, tier: MemoryTier = MemoryTier.PERSISTENT, source: MemorySource = MemorySource.UNKNOWN, ttl_seconds: float | None = None) -> MemoryEntry:
        entry = super().add(text, metadata, tier=tier, source=source, ttl_seconds=ttl_seconds)
        self._persist()
        return entry

    def clear(self, scope: dict[str, str] | None = None, *, tiers: set[MemoryTier] | None = None) -> None:
        super().clear(scope, tiers=tiers)
        self._persist_or_remove_if_empty()

    def delete(self, entry_id: str) -> bool:
        deleted = super().delete(entry_id)
        if deleted:
            self._persist_or_remove_if_empty()
        return deleted

    def purge_expired(self) -> int:
        purged = super().purge_expired()
        if purged:
            self._persist_or_remove_if_empty()
        return purged

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryError(f"Cannot load memory from {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or "entries" not in payload:
            raise MemoryError(
                f"Cannot load memory from {self.path}: not a memory file"
            )
        schema_version = payload.get("schema_version")
        if schema_version not in {1, SCHEMA_VERSION}:
            raise MemoryError(
                f"Cannot load memory from {self.path}: schema version "
                f"{schema_version!r} is not supported (expected 1 or {SCHEMA_VERSION})"
            )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise MemoryError(f"Cannot load memory from {self.path}: entries not a list")
        with self._lock:
            self._entries = [
                MemoryEntry(
                    id=str(item.get("id", "")), text=str(item.get("text", "")),
                    metadata=dict(item.get("metadata") or {}), created_at=str(item.get("created_at", "")),
                    tier=cast(MemoryTier, _enum_or_default(MemoryTier, item.get("tier"), MemoryTier.PERSISTENT)),
                    source=cast(MemorySource, _enum_or_default(MemorySource, item.get("source"), MemorySource.UNKNOWN)),
                    expires_at=_optional_string(item.get("expires_at")),
                )
                for item in entries
                if isinstance(item, dict)
            ]

    def _persist(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": [asdict(e) for e in self.all(include_expired=True)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _persist_or_remove_if_empty(self) -> None:
        if self.all(include_expired=True):
            self._persist()
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _enum_or_default(enum_type: type[MemoryTier] | type[MemorySource], value: object, default: MemoryTier | MemorySource) -> MemoryTier | MemorySource:
    try:
        return enum_type(value) if isinstance(value, str) else default
    except ValueError:
        return default


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None
