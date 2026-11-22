"""Structured output: parse/validate/correct the final answer end-to-end."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.structured import (
    check_final_answer,
    parse_structured,
    schema_instruction,
    validate_structured,
)
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response
from tests._helpers import EchoTool

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
}


def test_parse_structured_whole_text():
    ok, value = parse_structured('{"answer": "42"}')
    assert ok and value == {"answer": "42"}


def test_parse_structured_with_prose_prefix():
    ok, value = parse_structured('Here you go: {"answer": "42"} done')
    assert ok and value == {"answer": "42"}


def test_parse_structured_not_json():
    ok, value = parse_structured("no json here at all")
    assert not ok and value is None


def test_validate_structured_reports_path():
    problems = validate_structured(SCHEMA, {"confidenc": 0.5})
    assert problems  # missing required 'answer'
    assert any("answer" in p for p in problems)


def test_check_final_answer_combines():
    ok, value, problems = check_final_answer(SCHEMA, '{"answer": "42"}')
    assert ok and value["answer"] == "42" and not problems
    ok, value, problems = check_final_answer(SCHEMA, "nope")
    assert not ok and problems


def test_schema_instruction_contains_schema():
    instruction = schema_instruction(SCHEMA)
    assert "required" in instruction and "answer" in instruction


def test_agent_structured_success_first_try():
    llm = MockLLMClient([text_response('{"answer": "42", "confidence": 0.9}')])
    events: list[str] = []
    bus = EventBus()
    bus.subscribe("*", lambda e: events.append(e.type))
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    result = agent.run("what?", output_schema=SCHEMA)
    assert agent.structured_output == {"answer": "42", "confidence": 0.9}
    assert "confidence" in result
    assert "agent.structured_output_parsed" in events
    assert "agent.structured_output_retry" not in events


def test_agent_structured_correction_round():
    llm = MockLLMClient(
        [
            text_response("let me think... actually the answer"),
            text_response('{"answer": "42"}'),
        ]
    )
    bus = EventBus()
    events: list[str] = []
    bus.subscribe("*", lambda e: events.append(e.type))
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    agent.run("what?", output_schema=SCHEMA)
    assert agent.structured_output == {"answer": "42"}
    assert "agent.structured_output_retry" in events
    assert "agent.structured_output_invalid" not in events
    # The correction turn became a user message in the history.
    assert any(
        m.get("role") == "user" and "corrected JSON" in str(m.get("content"))
        for m in agent.state.messages
    )


def test_agent_structured_gives_up_after_retries():
    llm = MockLLMClient([text_response("junk1"), text_response("junk2"), text_response("junk3")])
    bus = EventBus()
    events: list[str] = []
    bus.subscribe("*", lambda e: events.append(e.type))
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    result = agent.run("what?", output_schema=SCHEMA)
    assert agent.structured_output is None
    assert result == "junk3"
    assert events.count("agent.structured_output_retry") == 2
    assert "agent.structured_output_invalid" in events


def test_agent_structured_retries_override():
    llm = MockLLMClient([text_response("junk")])
    agent = Agent(llm=llm, permission_policy=PermissivePolicy())
    agent.run("what?", output_schema=SCHEMA, structured_retries=0)
    # Zero retries: the single bad answer fails immediately.
    assert agent.structured_output is None


def test_agent_tools_then_structured_final():
    llm = MockLLMClient(
        [
            # one tool turn, then a schema-valid final answer
            __import__("kinetic_sdk.testing", fromlist=["tool_response"]).tool_response(
                "c1", "echo", {"message": "hi"}
            ),
            text_response('{"answer": "done", "confidence": 1}'),
        ]
    )
    agent = Agent(
        llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy()
    )
    agent.run("echo hi", output_schema=SCHEMA)
    assert agent.structured_output == {"answer": "done", "confidence": 1}


def test_run_without_schema_untouched():
    llm = MockLLMClient([text_response("plain text, not json")])
    agent = Agent(llm=llm, permission_policy=PermissivePolicy())
    result = agent.run("hi")
    assert result == "plain text, not json"
    assert agent.structured_output is None


# --- pseudo-tool (respond) submission -----------------------------------

def _respond(call_id: str, arguments: dict) -> object:
    from kinetic_sdk.llm.client import LLMResponse, ToolCall

    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name="respond", arguments=arguments)],
        stop_reason="tool_use",
    )


def test_structured_answer_submitted_via_respond_tool() -> None:
    llm = MockLLMClient([_respond("r1", {"answer": "42", "confidence": 0.9})])
    agent = Agent(llm, permission_policy=PermissivePolicy())

    assert agent.run("x", output_schema=SCHEMA) == '{"answer": "42", "confidence": 0.9}'
    assert agent.structured_output == {"answer": "42", "confidence": 0.9}
    # The respond call got a tool_result so history stays replay-valid.
    assert agent.state.messages[-1]["role"] == "user"
    assert agent.state.messages[-1]["content"][0]["type"] == "tool_result"


def test_respond_tool_appears_in_schemas_only_during_schema_run() -> None:
    agent = Agent(MockLLMClient([]), permission_policy=PermissivePolicy())
    agent.run("x", output_schema=SCHEMA, structured_retries=0)
    assert "respond" not in [tool["name"] for tool in agent.tool_schemas()]


def test_structured_respond_invalid_triggers_correction_then_succeeds() -> None:
    llm = MockLLMClient([_respond("r1", {"evil": True}), _respond("r2", {"answer": "ok"})])
    agent = Agent(llm, permission_policy=PermissivePolicy())

    result = agent.run("x", output_schema=SCHEMA, structured_retries=2)

    assert result == '{"answer": "ok"}'
    assert agent.structured_output == {"answer": "ok"}
    assert len(llm.calls) == 2
    history = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in agent.state.messages
    )
    assert "not satisfy" in history


def test_scalar_output_schema_keeps_text_path() -> None:
    llm = MockLLMClient([text_response("42")])
    agent = Agent(llm, permission_policy=PermissivePolicy())
    assert agent.run("x", output_schema={"type": "integer"}, structured_retries=1) == "42"
    assert agent.structured_output == 42


def test_respond_answer_passes_through_answer_verifier() -> None:
    from kinetic_sdk.agent.verifier import LLMAnswerVerifier

    judge = MockLLMClient([text_response("ACCEPTED: fine")])
    agent = Agent(
        MockLLMClient([_respond("r1", {"answer": "42"})]),
        permission_policy=PermissivePolicy(),
        answer_verifier=LLMAnswerVerifier(judge),
    )
    assert agent.run("x", output_schema=SCHEMA) == '{"answer": "42"}'
    assert len(judge.calls) == 1


import pytest


@pytest.mark.asyncio
async def test_async_structured_answer_via_respond_tool() -> None:
    from kinetic_sdk.agent.async_agent import AsyncAgent
    from kinetic_sdk.testing import AsyncMockLLMClient

    llm = AsyncMockLLMClient([_respond("r1", {"answer": "async"})])
    agent = AsyncAgent(llm, permission_policy=PermissivePolicy())
    assert await agent.run("x", output_schema=SCHEMA) == '{"answer": "async"}'
    assert agent.structured_output == {"answer": "async"}
