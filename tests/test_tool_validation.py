"""Tests for tool-input schema validation and its wiring into the agent loop."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockTool
from kinetic_sdk.tool.validation import validate_tool_input
from tests._helpers import EchoTool, MockLLM, text_response, tool_response

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
        "nested": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    },
    "required": ["name"],
}


def test_valid_input_passes():
    params = {
        "name": "a",
        "count": 3,
        "ratio": 0.5,
        "flag": True,
        "tags": ["x"],
        "mode": "fast",
        "nested": {"x": 1},
    }
    assert validate_tool_input(SCHEMA, params) == []


def test_missing_required():
    assert validate_tool_input(SCHEMA, {}) == ["missing required parameter 'name'"]


def test_wrong_types():
    problems = validate_tool_input(SCHEMA, {"name": 42, "count": "three", "flag": "yes"})
    assert any("name: expected string" in p for p in problems)
    assert any("count: expected integer" in p for p in problems)
    assert any("flag: expected boolean" in p for p in problems)


def test_bool_is_not_an_integer():
    assert validate_tool_input(SCHEMA, {"name": "a", "count": True}) != []
    assert validate_tool_input(SCHEMA, {"name": "a", "ratio": 2}) == []  # int ok as number


def test_unexpected_parameter_rejected():
    problems = validate_tool_input(SCHEMA, {"name": "a", "nmae": "typo"})
    assert any("unexpected parameter 'nmae'" in p for p in problems)


def test_unexpected_allowed_when_additional_properties_true():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "additionalProperties": True}
    assert validate_tool_input(schema, {"a": "x", "extra": 1}) == []


def test_no_properties_declared_accepts_anything():
    assert validate_tool_input({"type": "object"}, {"whatever": 1}) == []
    assert validate_tool_input({}, {"whatever": 1}) == []


def test_enum_rejected_value():
    problems = validate_tool_input(SCHEMA, {"name": "a", "mode": "ludicrous"})
    assert any("not in enum" in p for p in problems)


def test_array_items_validated():
    problems = validate_tool_input(SCHEMA, {"name": "a", "tags": ["ok", 5]})
    assert any("tags[1]: expected string" in p for p in problems)


def test_nested_object_validated():
    problems = validate_tool_input(SCHEMA, {"name": "a", "nested": {"x": "no"}})
    assert any("nested.x: expected integer" in p for p in problems)
    problems = validate_tool_input(SCHEMA, {"name": "a", "nested": {}})
    assert any("nested.x: missing required property" in p for p in problems)


def test_non_dict_params():
    assert validate_tool_input(SCHEMA, "a string") != []
    assert validate_tool_input(SCHEMA, None) != []


def test_non_object_root_schema_is_skipped():
    assert validate_tool_input({"type": "string"}, {"a": 1}) == []


# --- agent-loop wiring ---------------------------------------------------------


def test_agent_rejects_invalid_input_without_executing():
    tool = MockTool(
        name="strict",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
        result="ok",
    )
    llm = MockLLM(
        [
            tool_response("c1", "strict", {"n": "not-an-int"}),
            text_response("gave up"),
        ]
    )
    agent = Agent(llm=llm, tools=[tool], permission_policy=PermissivePolicy())

    assert agent.run("go") == "gave up"
    assert tool.calls == []  # never executed

    results = [
        b
        for m in agent.state.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert results[0]["is_error"] is True
    assert "invalid tool input" in results[0]["content"]
    assert "expected integer" in results[0]["content"]


def test_agent_executes_valid_input():
    tool = MockTool(
        name="strict",
        parameters={"type": "object", "properties": {"n": {"type": "integer"}},
                    "required": ["n"]},
        result="ok",
    )
    llm = MockLLM([tool_response("c1", "strict", {"n": 5}), text_response("done")])
    agent = Agent(llm=llm, tools=[tool], permission_policy=PermissivePolicy())

    assert agent.run("go") == "done"
    assert len(tool.calls) == 1


def test_validation_can_be_disabled():
    tool = MockTool(
        name="strict",
        parameters={"type": "object", "properties": {"n": {"type": "integer"}},
                    "required": ["n"]},
        result="ok",
    )
    llm = MockLLM([tool_response("c1", "strict", {"bogus": 1}), text_response("done")])
    agent = Agent(
        llm=llm,
        tools=[tool],
        permission_policy=PermissivePolicy(),
        validate_tool_inputs=False,
    )

    assert agent.run("go") == "done"
    assert len(tool.calls) == 1


def test_echo_tool_still_works_with_validation():
    """The classic happy path: valid args pass through untouched."""
    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "hello"}), text_response("fin")]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    assert agent.run("go") == "fin"
