"""Unit tests for the agent tool-calling loop, using a mock LLM + echo tool.

These cover the end-to-end Stage 1 flow: agent receives a message -> asks the
LLM -> executes tool calls -> feeds results back -> returns the final text.
Also covers error handling (unknown tool, tool raising), max-iteration safety,
event emission, and FLASH->MAX escalation.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.security.policy import PermissivePolicy
from tests._helpers import EchoTool, FailingTool, MockLLM, text_response, tool_response


def test_agent_returns_final_text_without_tool_calls():
    llm = MockLLM([text_response("The answer is 42.")])
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    result = agent.run("What is the answer?")
    assert result == "The answer is 42."
    # One LLM call, no tool executions.
    assert len(llm.calls) == 1


def test_agent_runs_tool_then_finishes():
    # Turn 1: model requests echo("hello").
    # Turn 2: model sees the echoed result and returns final text.
    llm = MockLLM(
        [
            tool_response("call_1", "echo", {"message": "hello"}),
            text_response("echoed: hello"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    result = agent.run("echo hello")
    assert result == "echoed: hello"
    assert len(llm.calls) == 2
    # The tool result must have been fed back to the LLM on turn 2.
    second_call_messages = llm.calls[1]["messages"]
    assert any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
        for m in second_call_messages
    )


def test_agent_handles_unknown_tool_gracefully():
    llm = MockLLM(
        [
            tool_response("c1", "no_such_tool", {}),
            text_response("recovered"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    result = agent.run("call missing tool")
    assert result == "recovered"
    # The unknown-tool error should have been recorded as a tool_result error.
    msg = agent.state.messages
    tool_results = [
        b for m in msg if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_results and tool_results[0].get("is_error") is True


def test_agent_handles_tool_exception():
    llm = MockLLM(
        [
            tool_response("c1", "boom", {}),
            text_response("handled failure"),
        ]
    )
    agent = Agent(llm=llm, tools=[FailingTool()], permission_policy=PermissivePolicy())
    result = agent.run("trigger failure")
    assert result == "handled failure"
    tool_results = [
        b for m in agent.state.messages if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_results[0].get("is_error") is True


def test_agent_max_iterations_safety():
    # The model keeps requesting the same tool forever; the loop must stop.
    loop = tool_response("c", "echo", {"message": "x"})
    llm = MockLLM([loop] * 100)
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy(), max_iterations=3)
    events: list[Event] = []
    agent.event_bus.subscribe("agent.error", events.append)
    agent.run("loop forever")
    assert any(e.payload.get("reason") == "max_iterations" for e in events)


def test_agent_emits_lifecycle_events():
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": "hi"}),
            text_response("done"),
        ]
    )
    events: list[Event] = []
    bus = EventBus()
    for t in (
        "agent.run_started",
        "agent.turn_started",
        "agent.llm_response",
        "agent.tool_call_started",
        "agent.tool_call_finished",
        "agent.run_finished",
    ):
        bus.subscribe(t, events.append)
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy(), event_bus=bus)
    agent.run("hi")

    types_seen = [e.type for e in events]
    assert "agent.run_started" in types_seen
    assert "agent.run_finished" in types_seen
    assert types_seen.count("agent.tool_call_started") == 1
    assert types_seen.count("agent.tool_call_finished") == 1
    finished = [e for e in events if e.type == "agent.run_finished"][0]
    assert finished.payload["final_text"] == "done"


def test_agent_duplicate_tool_name_rejected():
    with pytest.raises(ValueError):
        Agent(llm=MockLLM([]), tools=[EchoTool(), EchoTool()])


def test_agent_add_tool_runtime():
    agent = Agent(llm=MockLLM([]), tools=[])
    agent.add_tool(EchoTool())
    assert any(s["name"] == "echo" for s in agent.tool_schemas())
    with pytest.raises(ValueError):
        agent.add_tool(EchoTool())


def test_agent_appends_user_message_when_provided():
    llm = MockLLM([text_response("ok")])
    state = ConversationState()
    agent = Agent(llm=llm, tools=[], state=state)
    agent.run("hello")
    assert state.messages[0] == {"role": "user", "content": "hello"}


def test_agent_uses_system_prompt_from_state():
    llm = MockLLM([text_response("ok")])
    state = ConversationState(system_prompt="be brief")
    agent = Agent(llm=llm, state=state)
    agent.run("hi")
    assert llm.calls[0]["system"] == "be brief"


def test_agent_assistant_history_has_tool_use_blocks():
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": "x"}),
            text_response("final"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    agent.run("go")
    # Find the assistant message containing the tool_use block.
    assistant_msgs = [m for m in agent.state.messages if m["role"] == "assistant"]
    tool_uses = [
        b for m in assistant_msgs if isinstance(m["content"], list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
    ]
    assert tool_uses and tool_uses[0]["name"] == "echo"


def test_agent_escalate_flash_to_max():
    llm = MockLLM([text_response("ok")])
    agent = Agent(llm=llm, tools=[])
    agent.mode = AgentMode.FLASH
    events: list[Event] = []
    agent.event_bus.subscribe("agent.escalated", events.append)
    assert agent.escalate() is True
    assert agent.mode is AgentMode.MAX
    assert events and events[0].payload["to"] == "max"
    # Second escalate must be a no-op.
    assert agent.escalate() is False


def test_tool_schemas_passed_to_llm():
    llm = MockLLM([text_response("ok")])
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    agent.run("hi")
    tools = llm.calls[0]["tools"]
    assert tools is not None
    assert tools[0]["name"] == "echo"



# --- tool execution timeout ---------------------------------------------------


def test_tool_timeout_returns_error_result_and_loop_continues():
    import time

    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool.base import ToolResult

    def slow_handler(**params):
        time.sleep(2.0)
        return ToolResult(output="too late")

    llm = MockLLM(
        [
            tool_response("call_1", "slow", {}),
            text_response("recovered"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="slow", handler=slow_handler)],
        permission_policy=PermissivePolicy(),
        tool_timeout=0.05,
    )

    assert agent.run("go") == "recovered"

    tool_results = [
        block
        for msg in agent.state.messages
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert "timed out" in tool_results[0]["content"]
    # The timeout is audit-logged like any other result.
    assert any(
        "timed out" in str(entry.get("result", {}).get("error", ""))
        for entry in agent.audit_logger.entries
    )


def test_tool_timeout_none_preserves_direct_execution():
    from kinetic_sdk.testing import MockTool

    llm = MockLLM([tool_response("call_1", "fast", {}), text_response("done")])
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="fast", result="instant")],
        permission_policy=PermissivePolicy(),
    )
    assert agent.run("go") == "done"
    assert agent._executor is None  # pool is only created when a timeout is set


def test_tool_timeout_rejects_nonpositive():
    from kinetic_sdk.testing import MockTool

    with pytest.raises(ValueError):
        Agent(
            llm=MockLLM([text_response("x")]),
            tools=[MockTool(name="t", result=1)],
            permission_policy=PermissivePolicy(),
            tool_timeout=0,
        )


# --- cooperative cancellation ---------------------------------------------------


def test_cancel_between_iterations_stops_run_and_emits_event():
    from kinetic_sdk.hooks.base import HookPoint
    from kinetic_sdk.hooks.registry import HookRegistry

    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "x"})] * 10  # would loop forever
    )
    bus = EventBus()
    cancelled_events: list[Event] = []
    bus.subscribe("agent.cancelled", cancelled_events.append)
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        hooks=HookRegistry(),
        event_bus=bus,
    )

    def cancel_after_first_turn(ctx):
        if ctx.point is HookPoint.BEFORE_LLM_CALL and ctx.iteration == 1:
            agent.cancel()
        return None

    agent.hooks.register(HookPoint.BEFORE_LLM_CALL, cancel_after_first_turn)

    result = agent.run("loop")
    assert result == ""  # cancelled before a final answer existed
    assert len(cancelled_events) == 1
    # Turn 0 ran; the hook cancelled at turn 1 and the imminent LLM call was
    # skipped (the cancel check runs again right after hooks).
    assert len(llm.calls) == 1


def test_cancel_mid_tool_batch_skips_rest_but_keeps_history_valid():
    from kinetic_sdk.llm.client import LLMResponse, ToolCall
    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool.base import ToolResult

    holder: dict = {}

    def cancelling_handler(**params):
        holder["agent"].cancel()
        return ToolResult(output="first ran")

    llm = MockLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="first", arguments={}),
                    ToolCall(id="c2", name="second", arguments={}),
                ],
                stop_reason="tool_use",
            ),
            text_response("should not reach"),
        ]
    )
    second = MockTool(name="second", result="never")
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="first", handler=cancelling_handler), second],
        permission_policy=PermissivePolicy(),
    )
    holder["agent"] = agent

    agent.run("go")

    assert second.calls == []  # never executed
    # ...but its tool_use still got an error result: history stays valid.
    results = [
        b
        for m in agent.state.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert results[1]["is_error"] is True
    assert "cancelled" in results[1]["content"]


def test_cancel_flag_is_cleared_on_next_run():
    agent = Agent(
        llm=MockLLM([text_response("one"), text_response("two")]),
        tools=[],
        permission_policy=PermissivePolicy(),
    )
    agent.cancel()
    assert agent.cancelled is True
    # A cancelled flag from before must not poison a fresh run.
    assert agent.run("first") == "one"
    assert agent.cancelled is False


# --- parallel tool execution ----------------------------------------------------


def _multi_call_response():
    from kinetic_sdk.llm.client import LLMResponse, ToolCall

    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="c1", name="worker", arguments={"tag": "a", "delay": 0.15}),
            ToolCall(id="c2", name="worker", arguments={"tag": "b", "delay": 0.15}),
            ToolCall(id="c3", name="worker", arguments={"tag": "c", "delay": 0.15}),
        ],
        stop_reason="tool_use",
    )


def test_parallel_execution_actually_runs_concurrently():
    import threading
    import time

    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool.base import ToolResult

    active = 0
    max_active = 0
    lock = threading.Lock()

    def handler(tag, delay, **params):
        nonlocal max_active, active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(delay)
        with lock:
            active -= 1
        return ToolResult(output=f"done-{tag}")

    llm = MockLLM([_multi_call_response(), text_response("fin")])
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="worker", handler=handler)],
        permission_policy=PermissivePolicy(),
        parallel_tool_execution=True,
    )
    started = time.monotonic()
    assert agent.run("go") == "fin"
    elapsed = time.monotonic() - started

    assert max_active == 3  # all three were in flight at once
    assert elapsed < 0.45  # 3 x 0.15s sequential would exceed this


def test_parallel_results_appended_in_original_order():
    import time

    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool.base import ToolResult

    def handler(tag, delay, **params):
        time.sleep(delay)  # 'a' finishes LAST despite being first
        return ToolResult(output=f"done-{tag}")

    from kinetic_sdk.llm.client import LLMResponse, ToolCall

    llm = MockLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="worker", arguments={"tag": "a", "delay": 0.2}),
                    ToolCall(id="c2", name="worker", arguments={"tag": "b", "delay": 0.01}),
                ],
                stop_reason="tool_use",
            ),
            text_response("fin"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="worker", handler=handler)],
        permission_policy=PermissivePolicy(),
        parallel_tool_execution=True,
    )
    agent.run("go")

    results = [
        b
        for m in agent.state.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]  # original order


def test_parallel_policy_denial_still_applies_per_call():
    from kinetic_sdk.llm.client import LLMResponse, ToolCall
    from kinetic_sdk.security.policy import AllowListPolicy
    from kinetic_sdk.testing import MockTool

    llm = MockLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="allowed_tool", arguments={}),
                    ToolCall(id="c2", name="forbidden_tool", arguments={}),
                ],
                stop_reason="tool_use",
            ),
            text_response("fin"),
        ]
    )
    allowed = MockTool(name="allowed_tool", result="ran")
    forbidden = MockTool(name="forbidden_tool", result="should not run")
    agent = Agent(
        llm=llm,
        tools=[allowed, forbidden],
        permission_policy=AllowListPolicy(always_allow=["allowed_tool"]),
        parallel_tool_execution=True,
    )
    agent.run("go")

    assert len(allowed.calls) == 1
    assert forbidden.calls == []  # denied without executing
    results = [
        b
        for m in agent.state.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert results[1]["is_error"] is True
    assert "not in allow-list" in results[1]["content"]


def test_parallel_disabled_by_default_uses_sequential():
    import time

    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool.base import ToolResult

    def handler(tag, delay, **params):
        time.sleep(delay)
        return ToolResult(output=tag)

    llm = MockLLM([_multi_call_response(), text_response("fin")])
    agent = Agent(
        llm=llm,
        tools=[MockTool(name="worker", handler=handler)],
        permission_policy=PermissivePolicy(),
    )
    started = time.monotonic()
    agent.run("go")
    elapsed = time.monotonic() - started
    assert elapsed >= 0.45  # 3 x 0.15s strictly sequential
    assert agent._executor is None  # pool never created
