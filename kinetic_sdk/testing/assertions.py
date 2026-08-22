"""Assertion helpers for agent tests, built on :class:`RunTrace`.

These helpers read the structured events already collected by
:class:`~kinetic_sdk.observability.logger.InMemoryObservabilityLogger` — no
event parsing is reimplemented here. Typical usage::

    logger = InMemoryObservabilityLogger()
    agent = Agent(..., observability_logger=logger)
    agent.run("do something")
    trace = RunTrace.collect(logger.entries, agent.run_id)
    assert_tool_called(trace, "my_tool", times=1)
    assert_mode(trace, AgentMode.MAX)
    assert_no_permission_denied(trace)
"""

from __future__ import annotations

from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.observability.trace import RunTrace


def assert_tool_called(
    trace: RunTrace, tool_name: str, times: int | None = None
) -> None:
    """Assert *tool_name* ran during the traced run.

    With ``times=None`` at least one call is required; otherwise the count
    must match exactly.
    """
    calls = [c for c in trace.tool_calls() if c["name"] == tool_name]
    if times is None:
        assert calls, (
            f"Expected tool {tool_name!r} to be called at least once; "
            f"tools called: {[c['name'] for c in trace.tool_calls()]}"
        )
    else:
        assert len(calls) == times, (
            f"Expected tool {tool_name!r} to be called {times} time(s), "
            f"got {len(calls)}"
        )


def assert_mode(trace: RunTrace, expected_mode: AgentMode | str) -> None:
    """Assert the run ended in *expected_mode* (its final, post-escalation mode)."""
    expected = (
        expected_mode.value if isinstance(expected_mode, AgentMode) else str(expected_mode)
    )
    actual = trace.final_mode()
    assert actual == expected, f"Expected final mode {expected!r}, got {actual!r}"


def assert_no_permission_denied(trace: RunTrace) -> None:
    """Assert no tool call was denied by the permission policy during the run."""
    denied = [e for e in trace.events if e.get("event_type") == "security.permission_denied"]
    assert not denied, (
        f"Expected no permission denials, got {len(denied)}: "
        f"{[e['payload'].get('name') for e in denied]}"
    )
