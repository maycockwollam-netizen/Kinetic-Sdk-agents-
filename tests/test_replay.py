"""Tests for durable, non-executing agent replay playback."""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event
from kinetic_sdk.replay import (
    DeterministicReplayError,
    DeterministicToolReplay,
    JsonFileReplayStore,
    ReplayDebugger,
    ReplayDebugSession,
    ReplayForkError,
    ReplayRecorder,
    ReplayRun,
    ReplayStep,
    ReplayStoreError,
)
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response


def test_json_replay_store_round_trip_and_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    store = JsonFileReplayStore(path)
    replay = ReplayRun("run-1", [ReplayStep(0, "now", "agent.run_started", {})])
    replaced: list[tuple[str, str]] = []
    original_replace = __import__("os").replace

    def capture_replace(source, destination):
        replaced.append((str(source), str(destination)))
        original_replace(source, destination)

    monkeypatch.setattr("kinetic_sdk.replay.store.os.replace", capture_replace)
    store.save(replay)

    assert replaced and replaced[0][0].endswith(".tmp")
    assert store.load() == replay


def test_replay_store_rejects_bad_schema_and_invalid_steps(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 99, "replay": {}}), encoding="utf-8")
    with pytest.raises(ReplayStoreError, match="schema version"):
        JsonFileReplayStore(path).load()

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "replay": {
                    "run_id": "a",
                    "steps": [{"sequence": 4, "timestamp": "t", "event_type": "x", "payload": {}}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReplayStoreError, match="contiguous"):
        JsonFileReplayStore(path).load()


def test_agent_records_events_and_redacted_replay_valid_snapshot(tmp_path):
    store = JsonFileReplayStore(tmp_path / "run.json")
    recorder = ReplayRecorder(store)
    llm = MockLLMClient(
        [
            tool_response("call-1", "echo", {"message": "sk-super-secret-key-1234567890"}),
            text_response("done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[MockTool("echo", result="sk-super-secret-key-1234567890")],
        permission_policy=PermissivePolicy(),
        replay_recorder=recorder,
    )

    assert agent.run("hello") == "done"
    replay = store.load()
    assert replay is not None
    types = [step.event_type for step in replay.steps]
    assert "agent.run_started" in types
    assert "agent.tool_call_started" in types
    assert "agent.tool_call_finished" in types
    assert "replay.snapshot" in types
    assert types[-1] == "agent.run_finished"
    snapshot = next(step for step in replay.steps if step.event_type == "replay.snapshot")
    assert "sk-super-secret-key-1234567890" not in str(snapshot.payload)
    assert "[REDACTED]" in str(snapshot.payload)


def test_replay_recorder_skips_reasoning_trace_by_default(tmp_path):
    store = JsonFileReplayStore(tmp_path / "run.json")
    recorder = ReplayRecorder(store)

    recorder.handle(Event("agent.run_started", {"run_id": "run-1"}))
    recorder.handle(
        Event(
            "agent.reasoning_trace",
            {"run_id": "run-1", "reasoning": "private reasoning"},
        )
    )

    replay = store.load()
    assert replay is not None
    assert [step.event_type for step in replay.steps] == ["agent.run_started"]


def test_replay_recorder_captures_and_redacts_reasoning_trace_when_enabled(tmp_path):
    store = JsonFileReplayStore(tmp_path / "run.json")
    recorder = ReplayRecorder(store, capture_reasoning=True)

    recorder.handle(
        Event(
            "agent.reasoning_trace",
            {"run_id": "run-1", "api_key": "sk-super-secret-key-1234567890"},
        )
    )

    replay = store.load()
    assert replay is not None
    assert len(replay.steps) == 1
    assert replay.steps[0].event_type == "agent.reasoning_trace"
    assert replay.steps[0].payload["api_key"] == "[REDACTED]"


def test_debugger_moves_without_executing_or_mutating_replay(tmp_path):
    store = JsonFileReplayStore(tmp_path / "run.json")
    original = ReplayRun(
        "run-1",
        [
            ReplayStep(0, "t0", "agent.run_started", {}),
            ReplayStep(1, "t1", "agent.run_finished", {"final_text": "ok"}),
        ],
    )
    store.save(original)

    debugger = ReplayDebugger.open(store)
    assert debugger.current() is None
    assert debugger.next() == original.steps[0]
    assert debugger.seek(1) == original.steps[1]
    assert debugger.previous() == original.steps[0]
    assert debugger.seek(2) is None
    with pytest.raises(IndexError):
        debugger.seek(3)
    assert store.load() == original


def _raw_snapshot(sequence: int, messages: list[dict]) -> ReplayStep:
    return ReplayStep(
        sequence,
        "t",
        "replay.snapshot",
        {"state": {"system_prompt": None, "messages": messages, "max_messages": None, "metadata": {}}},
    )


def test_debug_session_forks_from_latest_snapshot_and_allows_input_model_and_tool_swap():
    replay = ReplayRun(
        "parent",
        [
            ReplayStep(0, "t0", "agent.run_started", {}),
            _raw_snapshot(1, [{"role": "user", "content": "old input"}]),
            ReplayStep(2, "t2", "agent.run_finished", {"final_text": "old"}),
        ],
    )
    session = ReplayDebugSession(replay)

    branch = session.fork(2).replace_input("new input")
    agent = branch.create_agent(
        MockLLMClient([text_response("new model answer")]),
        tools=[MockTool("replacement", result="unused")],
        permission_policy=PermissivePolicy(),
    )

    assert agent.run() == "new model answer"
    assert agent.state.messages[0]["content"] == "new input"
    assert branch.parent_run_id == "parent"
    assert branch.fork_sequence == 1


def test_branch_recorder_persists_parent_lineage(tmp_path):
    replay = ReplayRun("parent", [_raw_snapshot(0, [{"role": "user", "content": "old"}])])
    branch = ReplayDebugSession(replay).fork()
    store = JsonFileReplayStore(tmp_path / "child.json")
    recorder = branch.create_recorder(store, capture_raw_snapshots=True)
    agent = branch.create_agent(
        MockLLMClient([text_response("child")]),
        replay_recorder=recorder,
    )

    assert agent.run() == "child"
    child = store.load()
    assert child is not None
    assert (child.parent_run_id, child.fork_sequence) == ("parent", 0)


def test_debug_session_rejects_redacted_or_non_checkpoint_forks_and_builds_ui_timeline():
    replay = ReplayRun(
        "parent",
        [
            ReplayStep(0, "t0", "agent.tool_call_started", {"name": "echo"}),
            _raw_snapshot(1, [{"role": "user", "content": "[REDACTED]"}]),
        ],
    )
    session = ReplayDebugSession(replay)
    assert session.timeline()[0].label == "Tool started: echo"
    assert '"label": "Tool started: echo"' in session.timeline_json()
    with pytest.raises(ReplayForkError, match="redacted"):
        session.fork()
    with pytest.raises(ReplayForkError, match="no replay-valid"):
        session.fork(0)


def test_debug_session_compares_runs_without_timestamp_noise():
    left = ReplayRun("left", [ReplayStep(0, "one", "agent.run_started", {}), ReplayStep(1, "two", "agent.run_finished", {"final_text": "a"})])
    right = ReplayRun("right", [ReplayStep(0, "other", "agent.run_started", {}), ReplayStep(1, "later", "agent.run_finished", {"final_text": "b"})])

    diff = ReplayDebugSession(left).compare(right)

    assert [entry.kind for entry in diff.entries] == ["equal", "changed"]
    assert not diff.is_equal


def test_debug_session_compares_runs_ignoring_payload_run_ids():
    left = ReplayRun(
        "left-run",
        [
            ReplayStep(
                0,
                "one",
                "agent.run_started",
                {"run_id": "left-uuid", "mode": "max"},
            )
        ],
    )
    right = ReplayRun(
        "right-run",
        [
            ReplayStep(
                0,
                "other",
                "agent.run_started",
                {"run_id": "right-uuid", "mode": "max"},
            )
        ],
    )

    diff = ReplayDebugSession(left).compare(right)

    assert diff.entries[0].kind == "equal"
    assert diff.is_equal


def test_deterministic_tools_repeat_recorded_results_without_executing_real_tools():
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "echo", "input": {"message": "hi"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "hi"}]},
    ]
    replay = ReplayRun("run", [_raw_snapshot(0, messages)])
    tool = DeterministicToolReplay.from_replay(replay).tools()[0]

    assert tool.execute(message="hi").output == "hi"
    assert tool.execute(message="hi").is_error


def test_deterministic_tools_require_protected_raw_snapshots():
    replay = ReplayRun("run", [_raw_snapshot(0, [{"role": "user", "content": "[REDACTED]"}])])
    with pytest.raises(DeterministicReplayError, match="capture_raw_snapshots"):
        DeterministicToolReplay.from_replay(replay)


def test_checkpoint_resume_keeps_run_id_state_and_records_resume_event(tmp_path):
    from kinetic_sdk.replay import CheckpointManager

    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    agent = Agent(
        MockLLMClient([text_response("first")]),
        permission_policy=PermissivePolicy(),
    )
    assert agent.run("remember this") == "first"
    checkpoint_id = manager.save(agent)

    resumed = manager.resume(
        checkpoint_id,
        MockLLMClient([text_response("continued")]),
        [],
        permission_policy=PermissivePolicy(),
    )

    assert resumed.run_id == checkpoint_id
    assert resumed.state.messages == agent.state.messages
    assert resumed.run() == "continued"
    assert resumed.run_id == checkpoint_id
    replay = manager._store.load()
    assert replay is not None
    assert "replay.resumed" in [step.event_type for step in replay.steps]
