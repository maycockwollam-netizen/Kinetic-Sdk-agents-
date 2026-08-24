"""Long-term memory providers: what survives across sessions.

Conversation persistence (``conversation/store.py``) resumes ONE run's
history; compaction keeps a long turn inside the context window. Neither
answers "what did we learn in the session three weeks ago that matters now?"
— that question is a memory layer's job, and every mature agent SDK ships
one shape of it (LangChain's memory + vector stores, CrewAI's knowledge
bases, the OpenAI Agents session+external-memory split).

The layer here is deliberately minimal and provider-neutral:

* :class:`MemoryProvider` ABC — ``add`` / ``search`` / ``all`` / ``clear``.
* :class:`~kinetic_sdk.memory.inmemory.InMemoryMemory` — zero-dependency
  keyword-overlap search (tests, demos, small deployments).
* :class:`~kinetic_sdk.memory.json_file.JsonFileMemory` — same search,
  persisted to one JSON file (the file is user data, treated like the
  conversation store: never redacted, protect at the FS level).

An embedding/vector implementation only has to satisfy the same ABC to slot
in (the SDK ships none — embeddings need a provider dependency the core
does not carry, same rule as ``litellm`` being optional).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Minimal English/Vietnamese stopword set — overlap scoring without it
#: ranks "the/and/và/của" above the content words.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "it", "its", "this", "that",
        "và", "của", "là", "cho", "với", "trong", "một", "các", "những", "được",
    }
)

_TOKEN_RE = re.compile(r"[0-9a-zA-Zà-ỹÀ-Ỹ_]+")


def tokens(text: str) -> set[str]:
    """Extract lowercase content tokens (Unicode letters/digits/_)."""
    out: set[str] = set()
    for match in _TOKEN_RE.findall(text.lower()):
        if match not in _STOPWORDS and len(match) > 1:
            out.add(match)
    return out


@dataclass(frozen=True)
class MemoryEntry:
    """One stored memory.

    Attributes:
        id: Provider-assigned id (uuid hex).
        text: The remembered content (e.g. a user/assistant exchange).
        metadata: Free-form tags (run_id, task kind, source, ...).
        created_at: ISO-8601 UTC creation timestamp.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp shared by providers."""
    return datetime.now(timezone.utc).isoformat()


class MemoryError(Exception):
    """Memory operations that cannot be completed (corrupt store, bad input)."""


class MemoryProvider(ABC):
    """Interface for long-term memory backends.

    Implementations must be idempotent-safe: ``search`` on an empty store
    returns an empty list, never raises. ``add`` returns the created entry
    so callers can correlate it with audit/observability ids.
    """

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Store *text* and return the created :class:`MemoryEntry`."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Return up to *limit* entries relevant to *query* (best first)."""

    @abstractmethod
    def all(self) -> list[MemoryEntry]:
        """Every stored entry, oldest first."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all stored entries."""


def relevance(query: str, entry_text: str) -> float:
    """Keyword-overlap score in [0, 1]: |query ∩ entry| / |query|.

    Deterministic, dependency-free, and good enough to rank short texts.
    Embedding providers substitute cosine similarity under the same
    contract (0 .. completely unrelated .. 1).
    """
    q = tokens(query)
    if not q:
        return 0.0
    e = tokens(entry_text)
    return len(q & e) / len(q)
