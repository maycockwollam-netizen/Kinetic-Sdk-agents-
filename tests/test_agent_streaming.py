"""Streaming mode for the agent loop: ``Agent.run(stream=True)``.

Covers: real-time ``agent.text_delta`` events, tool calling under streaming
(including a model that streams text first and only then requests a tool),
and the fallback to plain ``chat()`` for clients without streaming support.
"""

from __future__ import annotations

from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response


def _collect(bus: EventBus) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    bus.subscribe("*", lambda e: events.append({"type": e.type, **e.payload}))
    return events


def test_stream_emits_text_deltas_concatenating_to_final_answer():
    bus = EventBus()
    events = _collect(bus)
    llm = MockLLMClient([text_response("The answer is forty-two.")])
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())

    answer = agent.run("meaning of life?", stream=True)

    deltas = [e["delta"] for e in events if e["type"] == "agent.text_delta"]
    assert deltas, "expected agent.text_delta events in stream mode"
    assert len(deltas) > 1, "mock client chunks text; expect multiple deltas"
    assert "".join(deltas) == "The answer is forty-two."
    assert answer == "The answer is forty-two."


def test_no_deltas_without_stream_flag():
    bus = EventBus()
    events = _collect(bus)
    llm = MockLLMClient([text_response("plain")])
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())

    answer = agent.run("hi")

    assert answer == "plain"
    assert not [e for e in events if e["type"] == "agent.text_delta"]


def test_streaming_tool_call_still_executes():
    """Model streams no text and ends the turn with a tool call."""
    bus = EventBus()
    events = _collect(bus)
    llm = MockLLMClient([
        tool_response(name="calc", arguments={"expression": "1+1"}),
        text_response("It is 2."),
    ])
    calc = MockTool("calc", result="2")
    agent = Agent(
        llm=llm, tools=[calc], event_bus=bus, permission_policy=PermissivePolicy()
    )

    answer = agent.run("1+1?", stream=True)

    assert calc.calls == [{"expression": "1+1"}]
    assert answer == "It is 2."
    deltas = [e["delta"] for e in events if e["type"] == "agent.text_delta"]
    assert "".join(deltas) == "It is 2."


def test_streaming_text_before_tool_call_preserves_order():
    """The model may stream text first, then request a tool in the same turn.
    Deltas must arrive BEFORE the tool starts, and the text must be kept in
    the conversation history ahead of the tool_use block."""
    bus = EventBus()
    events = _collect(bus)
    llm = MockLLMClient([
        LLMResponse(
            content="Let me compute that.",
            tool_calls=[ToolCall(id="c1", name="calc", arguments={"expression": "2*3"})],
            stop_reason="tool_use",
        ),
        text_response("6"),
    ])
    calc = MockTool("calc", result="6")
    agent = Agent(
        llm=llm, tools=[calc], event_bus=bus, permission_policy=PermissivePolicy()
    )

    agent.run("2*3?", stream=True)

    types = [e["type"] for e in events]
    first_delta = types.index("agent.text_delta")
    tool_started = types.index("agent.tool_call_started")
    assert first_delta < tool_started
    assert calc.calls == [{"expression": "2*3"}]
    # history: assistant turn holds the streamed text block before tool_use
    assistant = agent.state.messages[1]
    assert assistant["content"][0] == {"type": "text", "text": "Let me compute that."}
    assert assistant["content"][1]["type"] == "tool_use"


def test_streaming_falls_back_when_client_has_no_streaming():
    """A client without chat_stream support must still work in stream mode."""

    class NoStreamClient(LLMClient):
        model = "no-stream"

        def chat(self, messages, tools=None, system=None, **kwargs):
            return LLMResponse(content="fallback answer", stop_reason="end_turn")

    agent = Agent(llm=NoStreamClient(), permission_policy=PermissivePolicy())
    answer = agent.run("hi", stream=True)
    assert answer == "fallback answer"


def test_mock_client_streaming_consumes_script_like_chat():
    """chat_stream and chat share one script: one entry per turn, and the
    done event carries the full scripted response including tool calls."""
    llm = MockLLMClient([
        LLMResponse(
            content="working on it",
            tool_calls=[ToolCall(id="c9", name="calc", arguments={})],
            stop_reason="tool_use",
        ),
    ])
    events = list(llm.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    assert events[-1].type == "done"
    final = events[-1].delta
    assert final.content == "working on it"
    assert final.tool_calls[0].id == "c9"
    assert "".join(e.delta for e in events if e.type == "text") == "working on it"
    # script consumed: a second stream gets the empty default end_turn
    events2 = list(llm.chat_stream(messages=[{"role": "user", "content": "again"}]))
    assert events2[-1].delta.stop_reason == "end_turn"
    assert events2[-1].delta.content == ""
