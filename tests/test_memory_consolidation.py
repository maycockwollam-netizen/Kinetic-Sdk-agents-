"""Tests for episodic->persistent memory consolidation."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.memory import (
    InMemoryMemory,
    LLMMemoryConsolidator,
    MemorySource,
    MemoryTier,
    consolidate_episodic,
)
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import AsyncMockLLMClient, MockLLMClient, text_response


class _AlwaysConsolidate:
    """Stub consolidator that always folds the batch into one summary."""

    def consolidate(self, entries, context):  # type: ignore[no-untyped-def]
        return "Summary of " + "; ".join(entry.text for entry in entries)


class _NeverConsolidate:
    """Stub consolidator that declines every batch."""

    def consolidate(self, entries, context):  # type: ignore[no-untyped-def]
        del entries, context
        return None


def test_consolidate_episodic_folds_into_persistent_and_deletes_episodic() -> None:
    memory = InMemoryMemory()
    memory.add("run one did X", tier=MemoryTier.EPISODIC, source=MemorySource.LLM_INFERENCE)
    memory.add("run two did Y", tier=MemoryTier.EPISODIC, source=MemorySource.LLM_INFERENCE)

    summary_id, count = consolidate_episodic(memory, _AlwaysConsolidate())

    assert count == 2
    assert summary_id is not None
    assert memory.all(tiers={MemoryTier.EPISODIC}) == []
    persistent = memory.all(tiers={MemoryTier.PERSISTENT})
    assert len(persistent) == 1
    assert persistent[0].id == summary_id
    assert persistent[0].metadata["consolidated"] is True
    assert persistent[0].metadata["source_count"] == 2
    assert "run one did X" in persistent[0].text and "run two did Y" in persistent[0].text


def test_consolidate_episodic_declining_or_missing_consolidator_is_noop() -> None:
    memory = InMemoryMemory()
    memory.add("e1", tier=MemoryTier.EPISODIC)

    assert consolidate_episodic(memory, _NeverConsolidate()) == (None, 0)
    assert len(memory.all(tiers={MemoryTier.EPISODIC})) == 1

    assert consolidate_episodic(memory, None) == (None, 0)
    assert len(memory.all(tiers={MemoryTier.EPISODIC})) == 1


def test_consolidate_episodic_respects_scope() -> None:
    memory = InMemoryMemory()
    memory.add("a", tier=MemoryTier.EPISODIC, metadata={"project": "one"})
    memory.add("b", tier=MemoryTier.EPISODIC, metadata={"project": "two"})

    summary_id, count = consolidate_episodic(memory, _AlwaysConsolidate(), scope={"project": "one"})

    assert count == 1
    assert summary_id is not None
    leftover = memory.all(tiers={MemoryTier.EPISODIC})
    assert [e.text for e in leftover] == ["b"]


def test_llm_consolidator_summarises_above_threshold() -> None:
    llm = MockLLMClient([text_response("Durable lesson: keep build green.")])
    consolidator = LLMMemoryConsolidator(llm, min_entries=2, max_tokens=100)
    entries = _episodic(["a", "b"])

    summary = consolidator.consolidate(entries, {})
    assert summary == "Durable lesson: keep build green."
    assert len(llm.calls) == 1


def test_llm_consolidator_skips_small_batches_and_none_replies() -> None:
    llm = MockLLMClient([])
    consolidator = LLMMemoryConsolidator(llm, min_entries=3)
    assert consolidator.consolidate(_episodic(["only"]), {}) is None
    assert llm.calls == []  # no model call wasted

    llm = MockLLMClient([text_response("NONE")])
    consolidator = LLMMemoryConsolidator(llm, min_entries=1)
    assert consolidator.consolidate(_episodic(["noise"]), {}) is None


def test_llm_consolidator_fails_soft_on_model_error() -> None:
    class ExplodingLLM:
        def chat(self, messages=None, tools=None, system=None, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("offline")

    consolidator = LLMMemoryConsolidator(ExplodingLLM(), min_entries=1)
    assert consolidator.consolidate(_episodic(["x"]), {}) is None


def _episodic(texts: list[str]):  # type: ignore[no-untyped-def]
    memory = InMemoryMemory()
    for text in texts:
        memory.add(text, tier=MemoryTier.EPISODIC)
    return memory.all(tiers={MemoryTier.EPISODIC})


def test_agent_consolidates_episodic_memory_after_each_run() -> None:
    memory = InMemoryMemory()
    agent = Agent(
        MockLLMClient([text_response("a"), text_response("b")]),
        permission_policy=PermissivePolicy(),
        memory=memory,
        memory_consolidator=_AlwaysConsolidate(),
    )
    assert agent.run("first") == "a"
    assert agent.run("second") == "b"

    assert memory.all(tiers={MemoryTier.EPISODIC}) == []
    persistent = memory.all(tiers={MemoryTier.PERSISTENT})
    assert len(persistent) == 2
    assert all(entry.metadata.get("consolidated") for entry in persistent)


def test_agent_without_consolidator_keeps_episodic_entries() -> None:
    memory = InMemoryMemory()
    agent = Agent(
        MockLLMClient([text_response("a")]),
        permission_policy=PermissivePolicy(),
        memory=memory,
    )
    agent.run("first")

    assert len(memory.all(tiers={MemoryTier.EPISODIC})) == 1
    assert memory.all(tiers={MemoryTier.PERSISTENT}) == []


@pytest.mark.asyncio
async def test_async_agent_consolidates_memory_after_run() -> None:
    memory = InMemoryMemory()
    agent = AsyncAgent(
        AsyncMockLLMClient([text_response("a"), text_response("b")]),
        permission_policy=PermissivePolicy(),
        memory=memory,
        memory_consolidator=_AlwaysConsolidate(),
    )
    await agent.run("first")
    await agent.run("second")

    assert memory.all(tiers={MemoryTier.EPISODIC}) == []
    assert len(memory.all(tiers={MemoryTier.PERSISTENT})) == 2
