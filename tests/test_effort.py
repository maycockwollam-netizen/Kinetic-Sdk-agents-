"""Tests for per-turn reasoning-effort adjustment."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.effort import AdaptiveEffortPolicy, Effort, EffortContext
from kinetic_sdk.security.policy import PermissivePolicy
from tests._helpers import EchoTool, MockLLM, text_response, tool_response


def _agent(*, model: str, policy, script):
    return Agent(
        llm=MockLLM(script, model=model),
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        effort_policy=policy,
    )


def test_adaptive_policy_effort_levels() -> None:
    policy = AdaptiveEffortPolicy()
    assert policy.effort_for(EffortContext(iteration=0, previous_had_tool_calls=False)) == Effort.HIGH
    assert policy.effort_for(EffortContext(iteration=3, previous_had_tool_calls=True)) == Effort.MEDIUM
    assert policy.effort_for(EffortContext(iteration=4, previous_had_tool_calls=False)) == Effort.MEDIUM


def test_effort_forwarded_to_openai_style_model() -> None:
    llm = MockLLM(
        [tool_response("c1", "echo", {"message": "hi"}), text_response("done")],
        model="openai/gpt-4o",
    )
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        effort_policy=AdaptiveEffortPolicy(),
    )
    agent.run("echo hi")

    # Turn 0 -> high; turn after tool calls -> medium.
    assert llm.calls[0]["kwargs"].get("reasoning_effort") == "high"
    assert llm.calls[1]["kwargs"].get("reasoning_effort") == "medium"


def test_effort_not_sent_for_unsupported_model() -> None:
    llm = MockLLM([text_response("done")], model="mock-model")
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        effort_policy=AdaptiveEffortPolicy(),
    )
    agent.run("hi")
    assert "reasoning_effort" not in llm.calls[0]["kwargs"]


def test_effort_off_by_default() -> None:
    llm = MockLLM([text_response("done")], model="openai/gpt-4o")
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy())
    agent.run("hi")
    assert "reasoning_effort" not in llm.calls[0]["kwargs"]


def test_effort_policy_returning_none_is_ignored() -> None:
    llm = MockLLM([text_response("done")], model="openai/gpt-4o")

    class NonePolicy:
        def effort_for(self, context):
            return None

    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        effort_policy=NonePolicy(),
    )
    agent.run("hi")
    assert "reasoning_effort" not in llm.calls[0]["kwargs"]


@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "gpt-4o"])
def test_supports_effort_detects_openai(model) -> None:
    llm = MockLLM([text_response("done")], model=model)
    agent = Agent(llm=llm, tools=[], permission_policy=PermissivePolicy())
    assert agent._supports_effort() is True