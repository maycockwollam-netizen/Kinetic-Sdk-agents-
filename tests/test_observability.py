"""Tests for the observability package: structured logging, run tracing,
and the run_id wiring in the agent loop.
"""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.classifier import Classification, TaskComplexity
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.observability import (
    ConsoleObservabilityLogger,
    InMemoryObservabilityLogger,
    RunTrace,
)
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.security.redact import REDACTED
from tests._helpers import EchoTool, FailingTool, MockLLM, text_response, tool_response

FAKE_GHP = "ghp_" + "a1B2c3D4" * 5  # 40 chars after the prefix


class _ScriptedClassifier:
    """A classifier stub returning a fixed complexity (FLASH routing test)."""

    alias = "kinetic-classifier-v1"

    def __init__(self, complexity: TaskComplexity) -> None:
        self._complexity = complexity

    def classify(self, task: str) -> Classification:
        return Classification(
            complexity=self._complexity,
            mode=self._complexity.to_mode(),
            confidence=0.9,
            rationale="test",
        )


def _make_agent(obs_logger: InMemoryObservabilityLogger | None, responses: list) -> Agent:
    return Agent(
        llm=MockLLM(responses),
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        observability_logger=obs_logger,
    )


def _entry(event_type: str, ts: str, run_id: str = "r1", **payload) -> dict:
    return {
        "timestamp": ts,
        "event_type": event_type,
        "run_id": run_id,
        "payload": {"run_id": run_id, **payload},
    }


# --- ObservabilityLogger ---------------------------------------------------


def test_inmemory_logger_receives_main_events_via_wildcard():
    obs = InMemoryObservabilityLogger()
    agent = _make_agent(obs, [tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    agent.run("hello")

    for event_type in (
        "agent.run_started",
        "agent.tool_call_started",
        "agent.tool_call_finished",
        "agent.run_finished",
    ):
        events = obs.get_events(event_type)
        assert len(events) == 1, f"missing {event_type}"
        entry = events[0]
        assert entry["event_type"] == event_type
        assert entry["timestamp"]
        assert entry["run_id"] == agent.run_id
        assert entry["payload"]["run_id"] == agent.run_id


def test_get_events_without_filter_returns_everything():
    obs = InMemoryObservabilityLogger()
    agent = _make_agent(obs, [text_response("done")])
    agent.run("hello")

    all_events = obs.get_events()
    assert len(all_events) == len(obs.entries)
    # No tool calls in this run, but lifecycle + classification must be there.
    types = {e["event_type"] for e in all_events}
    assert {"agent.classified", "agent.run_started", "agent.llm_response", "agent.run_finished"} <= types


def test_console_logger_prints_readable_line(capsys):
    bus = EventBus()
    ConsoleObservabilityLogger().attach(bus)
    bus.publish(Event(type="agent.run_started", payload={"mode": "max", "run_id": "r1"}))

    line = capsys.readouterr().out.strip()
    assert line.startswith("[")
    assert "] [agent.run_started] " in line
    assert '"mode": "max"' in line


def test_payload_secrets_are_redacted():
    obs = InMemoryObservabilityLogger()
    bus = EventBus()
    obs.attach(bus)
    bus.publish(
        Event(
            type="agent.tool_call_finished",
            payload={
                "output_preview": f"token is {FAKE_GHP}",
                "nested": {"api_key": "sk-" + "x" * 30},
            },
        )
    )

    entry = obs.entries[0]
    assert FAKE_GHP not in str(entry)
    assert "sk-" + "x" * 30 not in str(entry)
    assert entry["payload"]["output_preview"] == f"token is {REDACTED}"
    assert entry["payload"]["nested"]["api_key"] == REDACTED


def test_run_id_unique_per_run_and_consistent_within_run():
    obs = InMemoryObservabilityLogger()
    agent = _make_agent(obs, [text_response("one"), text_response("two")])

    agent.run("first")
    run_id_1 = agent.run_id
    agent.run("second")
    run_id_2 = agent.run_id

    assert run_id_1 is not None and run_id_2 is not None
    assert run_id_1 != run_id_2
    for entry in obs.entries:
        assert entry["run_id"] in (run_id_1, run_id_2)
        assert entry["payload"]["run_id"] == entry["run_id"]
    assert len(obs.get_events("agent.run_started")) == 2


def test_agent_without_observability_logger_runs_fine():
    agent = _make_agent(None, [tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    assert agent.observability_logger is None
    assert agent.run("hello") == "done"
    # run_id is still stamped for tracing even without a logger attached.
    assert agent.run_id is not None


# --- RunTrace --------------------------------------------------------------


def test_trace_duration_uses_event_timestamps():
    trace = RunTrace(
        run_id="r1",
        events=[
            _entry("agent.run_started", "2026-08-21T10:00:00+00:00"),
            _entry("agent.tool_call_finished", "2026-08-21T10:00:01+00:00", name="echo", id="c1", is_error=False),
            _entry("agent.run_finished", "2026-08-21T10:00:02.500000+00:00", mode="max"),
        ],
    )
    assert trace.duration() == 2.5


def test_trace_duration_zero_when_boundaries_missing():
    assert RunTrace(run_id="r1", events=[]).duration() == 0.0
    only_started = RunTrace(run_id="r1", events=[_entry("agent.run_started", "2026-08-21T10:00:00+00:00")])
    assert only_started.duration() == 0.0


def test_trace_collect_filters_by_run_id():
    entries = [
        _entry("agent.run_started", "2026-08-21T10:00:00+00:00", run_id="r1"),
        _entry("agent.run_started", "2026-08-21T11:00:00+00:00", run_id="r2"),
        _entry("agent.run_finished", "2026-08-21T10:00:01+00:00", run_id="r1", mode="max"),
    ]
    trace = RunTrace.collect(entries, "r1")
    assert trace.run_id == "r1"
    assert len(trace.events) == 2
    assert all(e["run_id"] == "r1" for e in trace.events)


def test_trace_summary_reflects_escalation_denial_and_compaction():
    trace = RunTrace(
        run_id="r1",
        events=[
            _entry("agent.classified", "2026-08-21T10:00:00+00:00", mode="flash", complexity="simple"),
            _entry("agent.run_started", "2026-08-21T10:00:00+00:00", mode="flash"),
            _entry("agent.escalated", "2026-08-21T10:00:01+00:00", **{"from": "flash", "to": "max"}),
            _entry("security.permission_denied", "2026-08-21T10:00:02+00:00", name="terminal", reason="denied"),
            _entry("context.compacted", "2026-08-21T10:00:03+00:00", messages_removed=4),
            _entry("agent.tool_call_finished", "2026-08-21T10:00:04+00:00", name="echo", id="c1", is_error=False),
            _entry("agent.tool_call_finished", "2026-08-21T10:00:05+00:00", name="terminal", id="c2", is_error=True),
            _entry("agent.run_finished", "2026-08-21T10:00:06+00:00", mode="max"),
        ],
    )
    summary = trace.to_summary()
    assert summary == {
        "run_id": "r1",
        "duration": 6.0,
        "tool_call_count": 2,
        "tool_calls_failed": 1,
        "final_mode": "max",
        "escalated": True,
        "permission_denied": True,
        "context_compacted": True,
    }


def test_trace_summary_quiet_run_flags_all_false():
    trace = RunTrace(
        run_id="r1",
        events=[
            _entry("agent.run_started", "2026-08-21T10:00:00+00:00", mode="max"),
            _entry("agent.run_finished", "2026-08-21T10:00:01+00:00", mode="max"),
        ],
    )
    summary = trace.to_summary()
    assert summary["escalated"] is False
    assert summary["permission_denied"] is False
    assert summary["context_compacted"] is False
    assert summary["tool_call_count"] == 0
    assert summary["final_mode"] == "max"


def test_trace_mode_transitions_from_real_escalating_run():
    obs = InMemoryObservabilityLogger()
    agent = Agent(
        llm=MockLLM([tool_response("c1", "boom", {}), text_response("recovered")]),
        tools=[FailingTool()],
        classifier=_ScriptedClassifier(TaskComplexity.SIMPLE),
        permission_policy=PermissivePolicy(),
        observability_logger=obs,
    )
    assert agent.run("do something") == "recovered"

    trace = RunTrace.collect(obs.entries, agent.run_id)
    transitions = trace.mode_transitions()
    assert [t["event"] for t in transitions] == ["classified", "escalated"]
    assert transitions[0]["mode"] == "flash"
    assert transitions[1]["from"] == "flash"
    assert transitions[1]["to"] == "max"

    summary = trace.to_summary()
    assert summary["escalated"] is True
    assert summary["final_mode"] == "max"
    assert summary["tool_calls_failed"] == 1
    assert summary["duration"] >= 0.0
