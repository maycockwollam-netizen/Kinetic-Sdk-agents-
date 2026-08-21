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
from kinetic_sdk.event.bus import EventBus, Event
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.tool.base import Tool
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
