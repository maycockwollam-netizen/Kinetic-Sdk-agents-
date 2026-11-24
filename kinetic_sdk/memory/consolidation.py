"""Episodic-memory consolidation (tier lifetime management).

Raw Q/A pairs are stored as :class:`MemoryTier.EPISODIC` after every run.
Left alone they accumulate as near-duplicate noise that bloats recall. This
module folds a batch of episodic entries into ONE durable
:class:`MemoryTier.PERSISTENT` summary — the memory analogue of the context
manager's summarising compaction, applied to long-term memory instead of a
conversation window.

The summariser is a SEPARATE :class:`LLMClient` judge (same rule as
``skills/vet``: never reuse the agent's actor model to grade content it is
about to consume). Any failure — an exception, an empty reply, too few
entries to bother — falls back to doing nothing
(:func:`consolidate_episodic` returns ``(None, 0)``), so memory management
never breaks a run.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.memory.provider import (
    MemoryEntry,
    MemoryProvider,
    MemorySource,
    MemoryTier,
)
from kinetic_sdk.security.redact import redact_value

logger = logging.getLogger(__name__)

#: Entries below this count are not worth an extra model call.
DEFAULT_MIN_ENTRIES = 3

_PROMPT = (
    "You are a memory consolidator. Below are episodic records from prior "
    "agent runs. Distil them into ONE summary of at most 2 sentences holding "
    "only durable facts and lessons worth recalling in future sessions. "
    "Ignore one-off noise, repetition and transient details. If nothing "
    "durable is present, reply with the single word NONE.\n\n{records}\n\n"
    "Summary:"
)


class MemoryConsolidator(Protocol):
    """Turn a batch of episodic entries into one persistent summary string.

    Return ``None`` when the batch is not worth consolidating (e.g. too few
    entries, or no durable content) — the caller then skips the fold.
    """

    def consolidate(
        self, entries: list[MemoryEntry], context: dict[str, Any]
    ) -> str | None: ...


class NoopMemoryConsolidator:
    """Deliberately does nothing — disables consolidation explicitly."""

    def consolidate(
        self, entries: list[MemoryEntry], context: dict[str, Any]
    ) -> str | None:
        del entries, context
        return None


class LLMMemoryConsolidator:
    """Summarise episodic batches with a separate, cheap LLM.

    Args:
        llm: Any :class:`~kinetic_sdk.llm.client.LLMClient` (dedicated model).
        min_entries: Skip batches smaller than this (default 3) — a single
            Q/A pair is already compact.
        max_tokens: Reply budget for the summary.
    """

    def __init__(
        self, llm: LLMClient, *, min_entries: int = DEFAULT_MIN_ENTRIES, max_tokens: int = 200
    ) -> None:
        if min_entries < 1:
            raise ValueError("min_entries must be >= 1")
        self._llm = llm
        self._min_entries = min_entries
        self._max_tokens = max_tokens

    def consolidate(
        self, entries: list[MemoryEntry], context: dict[str, Any]
    ) -> str | None:
        if len(entries) < self._min_entries:
            return None
        # Content is redacted before leaving the SDK (memory may embed
        # credentials from tool outputs) — same rule as summarising compaction.
        records = "\n".join(
            f"- {redact_value(entry.text)}" for entry in entries
        )
        prompt = _PROMPT.format(records=records)
        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - consolidation must fail soft
            logger.warning("Memory consolidation call failed: %s", type(exc).__name__)
            return None
        summary = (response.content or "").strip()
        if not summary or summary.upper() == "NONE":
            return None
        return summary


#: Serialises read -> summarise -> add -> delete so two concurrent
#: consolidations cannot both fold the same episodic batch into duplicates.
_consolidation_lock = threading.Lock()


def consolidate_episodic(
    provider: MemoryProvider,
    consolidator: MemoryConsolidator | None,
    *,
    scope: dict[str, str] | None = None,
) -> tuple[str | None, int]:
    """Fold a provider's episodic entries into ONE persistent summary.

    Args:
        provider: The memory backend.
        consolidator: Summariser; ``None`` (or a consolidator that declines)
            turns this into a no-op returning ``(None, 0)``.
        scope: Optional metadata filter selecting which episodic entries are
            consolidated (e.g. ``{"project": "x"}``).

    Returns:
        ``(summary_entry_id, consolidated_count)``; ``(None, 0)`` when nothing
        was folded.
    """
    if consolidator is None:
        return None, 0
    with _consolidation_lock:
        return _consolidate_locked(provider, consolidator, scope)


def _consolidate_locked(
    provider: MemoryProvider,
    consolidator: MemoryConsolidator,
    scope: dict[str, str] | None,
) -> tuple[str | None, int]:
    episodic = provider.all(tiers={MemoryTier.EPISODIC}, scope=scope)
    if not episodic:
        return None, 0
    summary = consolidator.consolidate(
        episodic, {"scope": scope or {}}
    )
    if summary is None or not summary.strip():
        return None, 0
    entry = provider.add(
        summary.strip(),
        metadata={
            "consolidated": True,
            "source_count": len(episodic),
            **(scope or {}),
        },
        tier=MemoryTier.PERSISTENT,
        source=MemorySource.LLM_INFERENCE,
    )
    for item in episodic:
        provider.delete(item.id)
    return entry.id, len(episodic)