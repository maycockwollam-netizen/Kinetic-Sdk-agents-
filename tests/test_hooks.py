"""Tests for the hooks package and its wiring into the agent loop."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.hooks import HookContext, HookPoint, HookRegistry, HookResult
from kinetic_sdk.security import PermissivePolicy
from tests._helpers import EchoTool, MockLLM, text_response, tool_response


def _context(point: HookPoint = HookPoint.BEFORE_RUN) -> HookContext:
    return HookContext(point=point)


# --- HookRegistry ----------------------------------------------------------


def test_trigger_runs_hooks_in_registration_order():
    calls: list[str] = []
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: calls.append("first"))
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: calls.append("second"))

    registry.trigger(HookPoint.BEFORE_RUN, _context())

    assert calls == ["first", "second"]


def test_registering_same_hook_twice_is_noop():
    calls: list[str] = []

    def hook(ctx: HookContext) -> None:
        calls.append("x")

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, hook)
    registry.register(HookPoint.BEFORE_RUN, hook)

    registry.trigger(HookPoint.BEFORE_RUN, _context())

    assert calls == ["x"]


def test_unregister_removes_hook():
    calls: list[str] = []

    def hook(ctx: HookContext) -> None:
        calls.append("x")

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, hook)
    registry.unregister(HookPoint.BEFORE_RUN, hook)
    registry.unregister(HookPoint.BEFORE_RUN, hook)  # absent: no-op

    assert registry.hooks_for(HookPoint.BEFORE_RUN) == []
    registry.trigger(HookPoint.BEFORE_RUN, _context())
    assert calls == []


def test_trigger_collects_non_none_results():
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: None)
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: HookResult(should_continue=False))
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: HookResult())

    results = registry.trigger(HookPoint.BEFORE_RUN, _context())

    assert [r.should_continue for r in results] == [False, True]


def test_trigger_with_no_hooks_returns_empty_list():
    assert HookRegistry().trigger(HookPoint.ON_ERROR, _context()) == []


def test_failing_hook_does_not_crash_and_others_still_run():
    calls: list[str] = []

    def bad(ctx: HookContext) -> None:
        raise RuntimeError("hook exploded")

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, bad)
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: calls.append("survivor"))

    results = registry.trigger(HookPoint.BEFORE_RUN, _context())

    assert calls == ["survivor"]
    assert results == []


def test_failing_hook_emits_hooks_error_on_event_bus():
    bus = EventBus()
    errors: list[Event] = []
    bus.subscribe("hooks.error", errors.append)

    def bad(ctx: HookContext) -> None:
        raise ValueError("nope")

    registry = HookRegistry(event_bus=bus)
    registry.register(HookPoint.AFTER_RUN, bad)
    registry.trigger(HookPoint.AFTER_RUN, _context(HookPoint.AFTER_RUN))

    assert len(errors) == 1
    assert errors[0].payload["point"] == "after_run"
    assert "ValueError: nope" in errors[0].payload["error"]


def test_hooks_error_payload_is_redacted():
    bus = EventBus()
    errors: list[Event] = []
    bus.subscribe("hooks.error", errors.append)
    secret = "ghp_" + "a1B2c3D4" * 5

    def bad(ctx: HookContext) -> None:
        raise RuntimeError(f"leaked {secret}")

    registry = HookRegistry(event_bus=bus)
    registry.register(HookPoint.BEFORE_RUN, bad)
    registry.trigger(HookPoint.BEFORE_RUN, _context())

    assert secret not in errors[0].payload["error"]
    assert "[REDACTED]" in errors[0].payload["error"]


# --- Agent wiring ----------------------------------------------------------


def _make_agent(hooks: HookRegistry, llm: MockLLM, **kwargs) -> Agent:
    return Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        hooks=hooks,
        **kwargs,
    )


def test_agent_fires_run_and_llm_hooks_with_context():
    seen: dict[str, HookContext] = {}
    registry = HookRegistry()
    for point in (
        HookPoint.BEFORE_RUN,
        HookPoint.AFTER_RUN,
        HookPoint.BEFORE_LLM_CALL,
        HookPoint.AFTER_LLM_CALL,
    ):
        registry.register(point, lambda ctx, p=point: seen.setdefault(p.value, ctx))

    llm = MockLLM([text_response("hello")])
    agent = _make_agent(registry, llm)
    assert agent.run("hi") == "hello"

    assert seen["before_run"].user_message == "hi"
    assert seen["before_run"].run_id == agent.run_id
    assert seen["after_run"].final_text == "hello"
    assert seen["before_llm_call"].iteration == 0
    assert seen["after_llm_call"].llm_response is not None
    assert seen["after_llm_call"].llm_response.content == "hello"


def test_agent_fires_tool_hooks_around_execution():
    order: list[str] = []
    contexts: dict[str, HookContext] = {}

    def before(ctx: HookContext) -> None:
        order.append("before")
        contexts["before"] = ctx

    def after(ctx: HookContext) -> None:
        order.append("after")
        contexts["after"] = ctx

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_TOOL_CALL, before)
    registry.register(HookPoint.AFTER_TOOL_CALL, after)

    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "ping"}), text_response("done")]
    )
    agent = _make_agent(registry, llm)
    assert agent.run("go") == "done"

    assert order == ["before", "after"]
    assert contexts["before"].tool_name == "echo"
    assert contexts["before"].tool_input == {"message": "ping"}
    assert contexts["after"].tool_result is not None
    assert contexts["after"].tool_result.output == "ping"


def test_before_tool_call_hook_can_cancel_execution():
    registry = HookRegistry()
    registry.register(
        HookPoint.BEFORE_TOOL_CALL, lambda ctx: HookResult(should_continue=False)
    )
    spy_calls: list[dict] = []
    registry.register(
        HookPoint.AFTER_TOOL_CALL, lambda ctx: spy_calls.append(ctx.tool_input)
    )

    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "ping"}), text_response("stopped")]
    )
    agent = _make_agent(registry, llm)
    assert agent.run("go") == "stopped"

    # Cancelled: the tool never ran, AFTER_TOOL_CALL never fired, and the
    # model saw an error tool_result explaining the cancellation.
    assert spy_calls == []
    tool_results = [
        b
        for m in agent.state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_results[0]["is_error"] is True
    assert "cancelled by a before_tool_call hook" in str(tool_results[0]["content"])


def test_before_tool_call_hook_can_replace_tool_input():
    registry = HookRegistry()
    registry.register(
        HookPoint.BEFORE_TOOL_CALL,
        lambda ctx: HookResult(modified_context={"tool_input": {"message": "rewritten"}}),
    )

    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "original"}), text_response("done")]
    )
    agent = _make_agent(registry, llm)
    agent.run("go")

    tool_results = [
        b
        for m in agent.state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_results[0]["content"] == "rewritten"


def test_agent_wires_its_bus_into_registry_without_one():
    bus = EventBus()
    errors: list[Event] = []
    bus.subscribe("hooks.error", errors.append)

    def bad(ctx: HookContext) -> None:
        raise RuntimeError("boom")

    registry = HookRegistry()  # no bus of its own
    registry.register(HookPoint.BEFORE_RUN, bad)
    llm = MockLLM([text_response("ok")])
    agent = _make_agent(registry, llm, event_bus=bus)

    assert agent.run("go") == "ok"  # failing hook did not crash the run
    assert len(errors) == 1


def test_on_error_hook_fires_when_run_raises():
    seen: list[HookContext] = []
    registry = HookRegistry()
    registry.register(HookPoint.ON_ERROR, lambda ctx: seen.append(ctx))

    def exploding(messages, tools, system):
        raise RuntimeError("llm down")

    llm = MockLLM([exploding])
    agent = _make_agent(registry, llm)
    with pytest.raises(RuntimeError, match="llm down"):
        agent.run("go")

    assert len(seen) == 1
    assert "RuntimeError: llm down" in seen[0].error


def test_agent_without_hooks_runs_normally():
    llm = MockLLM([tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    agent = Agent(
        llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy()
    )
    assert agent.hooks is None
    assert agent.run("go") == "done"
