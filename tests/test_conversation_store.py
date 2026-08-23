"""Tests for conversation persistence (ConversationStore + agent wiring).

Covers the JSON file store (round-trip, atomic write, corruption handling)
and the agent integration: resume at construction, persist per turn, and
best-effort behaviour when the store breaks mid-run.
"""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import (
    ConversationStore,
    ConversationStoreError,
    JsonFileConversationStore,
    state_from_dict,
    state_to_dict,
)
from kinetic_sdk.security.policy import PermissivePolicy
from tests._helpers import EchoTool, MockLLM, text_response, tool_response


def _sample_state() -> ConversationState:
    state = ConversationState(system_prompt="You are a coding agent.", max_messages=50)
    state.add_user_message("fix the bug")
    state.add_assistant(
        [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "c1", "name": "echo", "input": {"message": "hi"}},
        ]
    )
    state.add_tool_result("c1", "hi")
    state.metadata["session"] = "s-123"
    return state


# --- store round-trip ----------------------------------------------------------


def test_json_store_round_trip(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    assert store.load() is None

    state = _sample_state()
    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.system_prompt == state.system_prompt
    assert loaded.messages == state.messages
    assert loaded.max_messages == 50
    assert loaded.metadata == {"session": "s-123"}


def test_json_store_overwrites_previous_save(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    store.save(_sample_state())
    fresh = ConversationState(system_prompt="other")
    store.save(fresh)
    loaded = store.load()
    assert loaded is not None
    assert loaded.system_prompt == "other"
    assert loaded.messages == []


def test_json_store_delete(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    store.delete()  # no-op when nothing saved
    store.save(_sample_state())
    store.delete()
    assert store.load() is None


def test_json_store_corrupted_file_raises_not_silent(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text("{not json", encoding="utf-8")
    store = JsonFileConversationStore(path)
    with pytest.raises(ConversationStoreError):
        store.load()


def test_json_store_wrong_schema_version_raises(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text(json.dumps({"schema_version": 999, "state": {}}), encoding="utf-8")
    with pytest.raises(ConversationStoreError, match="schema version"):
        JsonFileConversationStore(path).load()


def test_json_store_not_a_conversation_file_raises(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text(json.dumps(["just", "a", "list"]), encoding="utf-8")
    with pytest.raises(ConversationStoreError):
        JsonFileConversationStore(path).load()


def test_state_from_dict_validates_shape():
    with pytest.raises(TypeError):
        state_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        state_from_dict({"messages": "not-a-list"})


def test_json_store_write_is_atomic(tmp_path, monkeypatch):
    """A crash mid-write must leave the previous save intact."""
    path = tmp_path / "conv.json"
    store = JsonFileConversationStore(path)
    store.save(_sample_state())
    before = path.read_bytes()

    def exploding_dump(*args, **kwargs):
        raise OSError("disk full mid-write")

    monkeypatch.setattr(json, "dump", exploding_dump)
    with pytest.raises(OSError):
        store.save(ConversationState(system_prompt="new"))
    monkeypatch.undo()

    assert path.read_bytes() == before
    assert store.load() is not None  # previous save survived
    # No orphan temp files left behind.
    assert list(tmp_path.glob("*.tmp")) == []


# --- agent wiring ---------------------------------------------------------------


def test_agent_resumes_state_from_store(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    store.save(_sample_state())

    agent = Agent(llm=MockLLM([text_response("ok")]), state_store=store)

    assert agent.state.system_prompt == "You are a coding agent."
    assert len(agent.state.messages) == 3


def test_agent_explicit_state_wins_over_store(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    store.save(_sample_state())
    explicit = ConversationState(system_prompt="explicit")

    agent = Agent(
        llm=MockLLM([text_response("ok")]), state=explicit, state_store=store
    )
    assert agent.state is explicit


def test_agent_persists_after_each_turn(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "hi"}), text_response("done")]
    )
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        state_store=store,
    )

    assert agent.run("hello") == "done"

    saved = store.load()
    assert saved is not None
    # user + assistant(tool_use) + tool_result + assistant(final text)
    assert len(saved.messages) == 4
    assert saved.messages[-1]["role"] == "assistant"


def test_agent_resumed_state_is_replay_valid(tmp_path):
    """A resumed history must never end on a dangling tool_use."""
    store = JsonFileConversationStore(tmp_path / "conv.json")
    llm = MockLLM([tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        state_store=store,
    ).run("hello")

    saved = store.load()
    assert saved is not None
    last_content = saved.messages[-1]["content"]
    if isinstance(last_content, list):
        assert not any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in last_content
        )


class _ExplodingStore(ConversationStore):
    def __init__(self) -> None:
        self.load_calls = 0

    def save(self, state: ConversationState) -> None:
        raise OSError("disk on fire")

    def load(self) -> ConversationState | None:
        self.load_calls += 1
        return None

    def delete(self) -> None:
        pass


def test_broken_store_never_kills_the_run(caplog):
    import logging

    llm = MockLLM([tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        state_store=_ExplodingStore(),
    )
    with caplog.at_level(logging.WARNING, logger="kinetic_sdk.agent.agent"):
        assert agent.run("hello") == "done"
    assert any("persist" in r.message for r in caplog.records)


def test_state_round_trip_helpers_directly():
    state = _sample_state()
    restored = state_from_dict(state_to_dict(state))
    assert restored.messages == state.messages
    assert restored.system_prompt == state.system_prompt
