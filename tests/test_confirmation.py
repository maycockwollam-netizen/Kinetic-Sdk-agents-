"""Tests for the Confirmation UX: ON_PERMISSION_CHECK hooks confirming or
declining tool calls the policy flagged with ``requires_confirmation=True``.
"""

from __future__ import annotations

from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.hooks import HookContext, HookPoint, HookRegistry, HookResult
from kinetic_sdk.security import AllowListPolicy
from kinetic_sdk.tool.base import Tool, ToolResult
from tests._helpers import MockLLM, text_response, tool_response


class SpyTool(Tool):
    """Counts executions so tests can prove whether the tool actually ran."""

    name = "spy"
    description = "Records how many times it was executed."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executions = 0

    def execute(self, **params: Any) -> ToolResult:
        self.executions += 1
        return ToolResult(output="executed")


def _make_agent(hooks: HookRegistry | None) -> tuple[Agent, SpyTool, list[Event]]:
    """An agent whose 'spy' tool requires confirmation for target='destroy'."""
    spy = SpyTool()
    policy = AllowListPolicy(
        always_allow=["spy"],
        require_confirmation_patterns={"spy": ["destroy"]},
    )
    llm = MockLLM(
        [tool_response("c1", "spy", {"target": "destroy"}), text_response("end")]
    )
    bus = EventBus()
    denied: list[Event] = []
    bus.subscribe("security.permission_denied", denied.append)
    agent = Agent(
        llm=llm, tools=[spy], permission_policy=policy, event_bus=bus, hooks=hooks
    )
    return agent, spy, denied


def test_no_hooks_configured_denies_as_before():
    agent, spy, denied = _make_agent(hooks=None)

    assert agent.run("go") == "end"
    assert spy.executions == 0
    assert len(denied) == 1
    assert "requires manual confirmation" in denied[0].payload["reason"]


def test_hooks_without_permission_check_hook_denies():
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: None)  # unrelated point
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1


def test_hook_returning_none_denies():
    seen: list[HookContext] = []
    registry = HookRegistry()
    registry.register(HookPoint.ON_PERMISSION_CHECK, lambda ctx: seen.append(ctx))
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1
    # The observer hook still received the full context.
    assert seen[0].tool_name == "spy"
    assert seen[0].tool_input == {"target": "destroy"}
    assert seen[0].permission_decision is not None
    assert seen[0].permission_decision.requires_confirmation is True


def test_hook_confirming_lets_tool_execute():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=True)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    assert agent.run("go") == "end"
    assert spy.executions == 1
    assert denied == []


def test_hook_declining_keeps_denial():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=False)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1


def test_one_confirming_hook_among_decliners_is_enough():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=False)
    )
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=True)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 1
    assert denied == []


def test_raising_hook_falls_back_to_safe_denial():
    bus_errors: list[Event] = []

    def bad(ctx: HookContext) -> HookResult:
        raise RuntimeError("ui crashed")

    registry = HookRegistry()
    registry.register(HookPoint.ON_PERMISSION_CHECK, bad)
    agent, spy, denied = _make_agent(hooks=registry)
    agent.event_bus.subscribe("hooks.error", bus_errors.append)

    agent.run("go")

    # A crashed confirmation UI must never silently allow the call.
    assert spy.executions == 0
    assert len(denied) == 1
    assert len(bus_errors) == 1
    assert bus_errors[0].payload["point"] == "on_permission_check"
