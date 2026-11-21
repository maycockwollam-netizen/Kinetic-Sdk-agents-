"""Tests for CLI plan mode (PlanGateTool + agent integration)."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.cli.main import PlanGateTool
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response


def test_plan_gate_approves_when_confirmed() -> None:
    tool = PlanGateTool(lambda plan: True)
    result = tool.execute(plan="1. do x")
    assert not result.is_error
    assert "approved" in result.output.lower()


def test_plan_gate_rejects_when_denied() -> None:
    tool = PlanGateTool(lambda plan: False)
    result = tool.execute(plan="bad plan")
    assert result.is_error
    assert "rejected" in str(result.error)


def test_plan_gate_requires_non_empty_plan() -> None:
    tool = PlanGateTool(lambda plan: True)
    assert tool.execute(plan="").is_error


def test_plan_gate_rejects_non_callable() -> None:
    import pytest

    with pytest.raises(TypeError):
        PlanGateTool(on_propose="nope")  # type: ignore[arg-type]


def test_agent_plan_gate_approval_flow() -> None:
    proposed: list[str] = []
    gate = PlanGateTool(lambda plan: proposed.append(plan) or True)
    llm = MockLLMClient(
        [tool_response("c1", "plan_gate", {"plan": "step one"}), text_response("done")]
    )
    agent = Agent(llm=llm, tools=[gate], permission_policy=PermissivePolicy())
    assert agent.run("plan the task") == "done"
    assert proposed == ["step one"]


def test_agent_plan_gate_rejection_loop() -> None:
    proposed: list[str] = []

    def review(plan: str) -> bool:
        proposed.append(plan)
        return len(proposed) >= 2  # accept on second proposal

    gate = PlanGateTool(review)
    llm = MockLLMClient(
        [
            tool_response("c1", "plan_gate", {"plan": "v1"}),
            tool_response("c2", "plan_gate", {"plan": "v2"}),
            text_response("ok"),
        ]
    )
    agent = Agent(llm=llm, tools=[gate], permission_policy=PermissivePolicy())
    assert agent.run("task") == "ok"
    assert proposed == ["v1", "v2"]