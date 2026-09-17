"""Production-oriented memory providers with tier, provenance, TTL and scope.

Unlike conversation persistence (one resumed history) or context compaction
(one long turn), this layer retains facts across runs.  It deliberately keeps
one provider contract while annotating each entry with its lifecycle:
working memory is run-local, episodic memory records a run, and persistent
memory survives sessions.  Providers also expose source provenance, optional
expiry, and subset-matched workspace/project/user scopes.  This is a smaller,
provider-neutral equivalent of the memory abstractions offered by LangChain
and CrewAI: it does not promise automatic fact extraction or a vector DB.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class MemoryTier(str, Enum):
    """Lifecycle class of a memory entry."""

    WORKING = "working"
    EPISODIC = "episodic"
    PERSISTENT = "persistent"


class MemorySource(str, Enum):
    """How the remembered content was obtained."""

    UNKNOWN = "unknown"  # Compatibility marker for pre-provenance data.
    USER_INPUT = "user_input"
    TOOL_RESULT = "tool_result"
    LLM_INFERENCE = "llm_inference"


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "it", "its", "this", "that",
    "và", "của", "là", "cho", "với", "trong", "một", "các", "những", "được",
})
_TOKEN_RE = re.compile(r"[0-9a-zA-Zà-ỹÀ-Ỹ_]+")


def tokens(text: str) -> set[str]:
    """Extract lowercase content tokens (Unicode letters/digits/_)."""
    return {match for match in _TOKEN_RE.findall(text.lower()) if match not in _STOPWORDS and len(match) > 1}


@dataclass(frozen=True)
class MemoryEntry:
    """One remembered value, including lifecycle and provenance metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    tier: MemoryTier = MemoryTier.PERSISTENT
    source: MemorySource = MemorySource.UNKNOWN
    expires_at: str | None = None


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp shared by providers."""
    return datetime.now(timezone.utc).isoformat()


def expiry_iso(ttl_seconds: float | None, created_at: str) -> str | None:
    """Return an expiry timestamp, validating TTL without hidden scheduling."""
    if ttl_seconds is None:
        return None
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or ttl_seconds < 0:
        raise MemoryError("ttl_seconds must be a non-negative number or None")
    return (datetime.fromisoformat(created_at) + timedelta(seconds=ttl_seconds)).isoformat()


def is_expired(entry: MemoryEntry, now: datetime | None = None) -> bool:
    """Whether *entry* is expired at *now* (malformed persisted expiry is expired)."""
    if entry.expires_at is None:
        return False
    try:
        expires_at = datetime.fromisoformat(entry.expires_at.replace("Z", "+00:00"))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return expires_at <= (now or datetime.now(timezone.utc))


def matches_scope(entry: MemoryEntry, scope: dict[str, str] | None) -> bool:
    """Return true when every requested scope key/value occurs in metadata."""
    return scope is None or all(entry.metadata.get(key) == value for key, value in scope.items())


class MemoryError(Exception):
    """Memory operations that cannot be completed (corrupt store, bad input)."""


class MemoryProvider(ABC):
    """Interface shared by keyword, JSON, and vector memory backends."""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None, *, tier: MemoryTier = MemoryTier.PERSISTENT, source: MemorySource = MemorySource.UNKNOWN, ttl_seconds: float | None = None) -> MemoryEntry:
        """Store text with lifecycle/provenance; ``UNKNOWN`` preserves old callers."""

    @abstractmethod
    def search(self, query: str, limit: int = 5, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        """Return relevant non-expired entries, optionally constrained by tier/scope."""

    @abstractmethod
    def all(self, *, tiers: set[MemoryTier] | None = None, scope: dict[str, str] | None = None, include_expired: bool = False) -> list[MemoryEntry]:
        """Return stored entries, oldest first, with optional lifecycle filters."""

    @abstractmethod
    def clear(self, scope: dict[str, str] | None = None, *, tiers: set[MemoryTier] | None = None) -> None:
        """Drop all entries matching optional scope and tiers."""

    def delete(self, entry_id: str) -> bool:
        """Delete one entry by id; legacy providers may opt out explicitly."""
        raise NotImplementedError("this memory provider does not support deletion by id")

    def purge_expired(self) -> int:
        """Delete expired entries; retained for backwards-compatible providers."""
        return 0


def relevance(query: str, entry_text: str) -> float:
    """Keyword-overlap score in [0, 1]: |query ∩ entry| / |query|."""
    query_tokens = tokens(query)
    return len(query_tokens & tokens(entry_text)) / len(query_tokens) if query_tokens else 0.0
