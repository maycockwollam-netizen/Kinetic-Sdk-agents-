"""Tests for the public testing utilities (mocks + assertion helpers).

Includes a self-test: a complete agent test written with nothing but the
public ``kinetic_sdk.testing`` API, mirroring the package docstring example.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.observability import InMemoryObservabilityLogger, RunTrace
from kinetic_sdk.security import AllowListPolicy, PermissivePolicy
from kinetic_sdk.testing import (
    MockLLMClient,
    MockTool,
    assert_mode,
    assert_no_permission_denied,
    assert_tool_called,
    text_response,
    tool_response,
)
from kinetic_sdk.tool.base import ToolResult


# --- MockLLMClient ---------------------------------------------------------


def test_mock_llm_replays_responses_in_order_and_records_calls():
    llm = MockLLMClient([text_response("one"), text_response("two")])

    first = llm.chat(messages=[{"role": "user", "content": "a"}])
    second = llm.chat(messages=[{"role": "user", "content": "b"}], system="s")

    assert first.content == "one"
    assert second.content == "two"
    assert len(llm.calls) == 2
    assert llm.calls[1]["system"] == "s"


def test_mock_llm_supports_callable_entries():
    def branch(messages, tools, system):
        return text_response(f"saw {len(messages)} messages")

    llm = MockLLMClient([branch])
    assert llm.chat(messages=[{"role": "user", "content": "x"}]).content == "saw 1 messages"


def test_mock_llm_returns_empty_end_turn_when_script_runs_out():
    llm = MockLLMClient([])
    response = llm.chat(messages=[])
    assert response.content == ""
    assert response.stop_reason == "end_turn"
    assert response.tool_calls == []


def test_response_builders():
    text = text_response("hi")
    assert text.content == "hi" and text.stop_reason == "end_turn"

    tool = tool_response("id-1", "calc", {"x": 1})
    assert tool.stop_reason == "tool_use"
    assert tool.tool_calls[0].id == "id-1"
    assert tool.tool_calls[0].name == "calc"
    assert tool.tool_calls[0].arguments == {"x": 1}


# --- MockTool --------------------------------------------------------------


def test_mock_tool_returns_fixed_result_and_records_calls():
    tool = MockTool("calc", result="42")
    result = tool.execute(expression="6*7")
    assert isinstance(result, ToolResult)
    assert result.output == "42"
    assert tool.calls == [{"expression": "6*7"}]


def test_mock_tool_wraps_toolresult_verbatim():
    tool = MockTool("boom", result=ToolResult(error="kaboom"))
    result = tool.execute()
    assert result.is_error
    assert result.error == "kaboom"


def test_mock_tool_handler_takes_params():
    tool = MockTool("adder", handler=lambda a, b: a + b)
    assert tool.execute(a=2, b=3).output == 5


def test_mock_tool_handler_may_return_toolresult():
    tool = MockTool("failer", handler=lambda **p: ToolResult(error="nope"))
    assert tool.execute().is_error


def test_mock_tool_rejects_result_and_handler_together():
    with pytest.raises(ValueError, match="either a fixed result or a handler"):
        MockTool("bad", result="x", handler=lambda: "y")


def test_mock_tool_default_output_and_schema():
    tool = MockTool("plain")
    assert tool.execute(a=1).output == {"tool": "plain", "params": {"a": 1}}
    schema = tool.to_schema()
    assert schema["name"] == "plain"
    assert schema["input_schema"] == {"type": "object", "properties": {}}


# --- Assertion helpers -----------------------------------------------------


def _run_traced_agent(llm: MockLLMClient, tools, policy) -> tuple[Agent, RunTrace]:
    obs = InMemoryObservabilityLogger()
    agent = Agent(
        llm=llm,
        tools=tools,
        permission_policy=policy,
        observability_logger=obs,
    )
    agent.run("go")
    return agent, RunTrace.collect(obs.entries, agent.run_id)


def test_assert_tool_called_pass_and_fail():
    llm = MockLLMClient(
        [tool_response("c1", "calc", {"x": 1}), text_response("done")]
    )
    _, trace = _run_traced_agent(llm, [MockTool("calc")], PermissivePolicy())

    assert_tool_called(trace, "calc")
    assert_tool_called(trace, "calc", times=1)
    with pytest.raises(AssertionError, match="at least once"):
        assert_tool_called(trace, "missing")
    with pytest.raises(AssertionError, match="2 time"):
        assert_tool_called(trace, "calc", times=2)


def test_assert_mode_pass_and_fail():
    llm = MockLLMClient([text_response("done")])
    _, trace = _run_traced_agent(llm, [], PermissivePolicy())

    assert_mode(trace, AgentMode.MAX)  # DefaultClassifier always routes MAX
    assert_mode(trace, "max")
    with pytest.raises(AssertionError, match="flash"):
        assert_mode(trace, AgentMode.FLASH)


def test_assert_no_permission_denied_pass_and_fail():
    ok_llm = MockLLMClient(
        [tool_response("c1", "calc", {}), text_response("done")]
    )
    _, ok_trace = _run_traced_agent(ok_llm, [MockTool("calc")], PermissivePolicy())
    assert_no_permission_denied(ok_trace)

    denied_llm = MockLLMClient(
        [tool_response("c1", "calc", {}), text_response("blocked")]
    )
    # Default-constructed AllowListPolicy denies everything.
    _, denied_trace = _run_traced_agent(denied_llm, [MockTool("calc")], AllowListPolicy())
    with pytest.raises(AssertionError, match="permission denials"):
        assert_no_permission_denied(denied_trace)


# --- Self-test: a full agent test using only the public testing API --------


def test_full_agent_flow_with_public_utilities():
    llm = MockLLMClient(
        [
            tool_response("call-1", "calculator", {"expression": "1+1"}),
            text_response("The answer is 2."),
        ]
    )
    calc = MockTool(
        "calculator",
        description="Evaluate an arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        result="2",
    )
    obs = InMemoryObservabilityLogger()
    agent = Agent(
        llm=llm,
        tools=[calc],
        permission_policy=PermissivePolicy(),
        observability_logger=obs,
    )

    answer = agent.run("What is 1+1?")

    trace = RunTrace.collect(obs.entries, agent.run_id)
    assert answer == "The answer is 2."
    assert calc.calls == [{"expression": "1+1"}]
    assert_tool_called(trace, "calculator", times=1)
    assert_mode(trace, "max")
    assert_no_permission_denied(trace)


def test_mock_tool_error_drives_agent_error_path():
    llm = MockLLMClient(
        [tool_response("c1", "boom", {}), text_response("recovered")]
    )
    obs = InMemoryObservabilityLogger()
    agent = Agent(
        llm=llm,
        tools=[MockTool("boom", result=ToolResult(error="kaboom"))],
        permission_policy=PermissivePolicy(),
        observability_logger=obs,
    )

    assert agent.run("go") == "recovered"
    trace = RunTrace.collect(obs.entries, agent.run_id)
    assert trace.to_summary()["tool_calls_failed"] == 1
