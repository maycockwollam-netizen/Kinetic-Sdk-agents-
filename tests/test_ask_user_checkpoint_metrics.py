"""AskUserTool, checkpoint helpers, metrics collector."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.ask_user import AskUserTool
from kinetic_sdk.conversation.checkpoint import (
    CheckpointError,
    fork_store,
    rewind_state,
)
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import JsonFileConversationStore
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.observability.metrics import MetricsCollector
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from tests._helpers import EchoTool

# --- AskUserTool --------------------------------------------------------


def test_ask_user_handler_answer():
    tool = AskUserTool(handler=lambda q: "blue" if "color" in q else "?")
    result = tool.execute(question="favorite color?")
    assert result.output == "blue"


def test_ask_user_default_cli_prompt(monkeypatch, capsys):
    tool = AskUserTool()
    import builtins

    monkeypatch.setattr(builtins, "input", lambda prompt="": "42")
    result = tool.execute(question="answer?")
    assert result.output == "42"


def test_ask_user_empty_question_rejected():
    assert AskUserTool().execute(question="   ").is_error


def test_ask_user_handler_error_is_tool_error():
    def boom(_q: str) -> str:
        raise RuntimeError("ui closed")

    result = AskUserTool(boom).execute(question="q")
    assert result.is_error and "ui closed" in result.error


def test_ask_user_end_to_end_in_loop():
    llm = MockLLMClient(
        [
            tool_response("c1", "ask_user", {"question": "confirm?"}),
            text_response("you said ok"),
        ]
    )
    tool = AskUserTool(handler=lambda q: "ok")
    agent = Agent(llm=llm, tools=[tool], permission_policy=PermissivePolicy())
    assert agent.run("confirm something") == "you said ok"


# --- Checkpoint ---------------------------------------------------------


def _state() -> ConversationState:
    return ConversationState(
        system_prompt="sys",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hey"}]},
            {"role": "user", "content": "run it"},
        ],
        metadata={"k": 1},
    )


def test_rewind_to_text_boundary():
    new = rewind_state(_state(), 2)
    assert len(new.messages) == 2
    assert new.system_prompt == "sys" and new.metadata == {"k": 1}


def test_rewind_rejects_dangling_tool_use():
    state = ConversationState(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "echo", "input": {}}
                ],
            },
        ],
    )
    with pytest.raises(CheckpointError):
        rewind_state(state, 2)


def test_rewind_allows_tool_result_tail():
    state = ConversationState(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "echo", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "x"}],
            },
            {"role": "assistant", "content": "done"},
        ],
    )
    new = rewind_state(state, 2)  # ends on the tool_result - fine
    assert len(new.messages) == 2


def test_fork_store_copies_saved_state(tmp_path):
    src_path = tmp_path / "a.json"
    src = JsonFileConversationStore(src_path)
    src.save(_state())
    dest = fork_store(src, tmp_path / "b.json")
    loaded = dest.load()
    assert loaded is not None and len(loaded.messages) == len(_state().messages)


def test_fork_store_refuses_existing_destination(tmp_path):
    src = JsonFileConversationStore(tmp_path / "a.json")
    target = tmp_path / "b.json"
    target.write_text("{}")
    with pytest.raises(CheckpointError):
        fork_store(src, target)


def test_fork_store_empty_source_ok(tmp_path):
    src = JsonFileConversationStore(tmp_path / "a.json")
    dest = fork_store(src, tmp_path / "b.json")
    assert dest.load() is None


# --- MetricsCollector ---------------------------------------------------


def test_metrics_counts_everything():
    bus = EventBus()
    metrics = MetricsCollector()
    metrics.attach(bus)
    llm = MockLLMClient([text_response("ok")])
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    agent.run("hi")
    snap = metrics.snapshot()
    assert snap["runs_started"] == 1 and snap["runs_finished"] == 1
    assert snap["event_counts"]["agent.run_started"] == 1
    assert snap["run_duration_max"] >= 0.0
    metrics.detach(bus)


def test_metrics_tracks_tool_failures_and_usage():
    bus = EventBus()
    metrics = MetricsCollector()
    metrics.attach(bus)
    responses = [
        tool_response("c1", "echo", {"message": "x"}),
        text_response("done"),
    ]
    llm = MockLLMClient(responses)
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        event_bus=bus,
        permission_policy=PermissivePolicy(),
    )
    agent.run("echo x")
    snap = metrics.snapshot()
    assert snap["tool_calls"] == 1 and snap["tool_failures"] == 0
