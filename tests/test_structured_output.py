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
