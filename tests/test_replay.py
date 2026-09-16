"""Tests for durable, non-executing agent replay playback."""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.replay import (
    JsonFileReplayStore,
    ReplayDebugger,
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
