"""Tests for the Workflow composition layer."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import SubagentSpec
from kinetic_sdk.testing import MockLLMClient, text_response
from kinetic_sdk.workflow import Workflow


def _spec(name: str = "worker") -> SubagentSpec:
    return SubagentSpec(name=name, system_prompt="p", description=f"{name} agent")


def test_workflow_runs_sequential_steps() -> None:
    llm = MockLLMClient([
        text_response("plan ok"),      # delegate child
        text_response("survey a"),     # map_reduce item a
        text_response("survey b"),     # map_reduce item b
        text_response("final summary"),  # run_agent
    ])
    parent = Agent(llm=llm, permission_policy=PermissivePolicy())
    wf = (
        Workflow(parent)
        .delegate("plan", _spec(), "plan")
        .map_reduce("survey", _spec(), ["a", "b"], max_concurrency=1)
        .run_agent("final", "summarize")
    )
    res = wf.run()
    assert res.outputs["plan"] == "plan ok"
    assert res.outputs["survey"] == "survey a\n\n--- Sub-agent result ---\n\nsurvey b"
    assert res.outputs["final"] == "final summary"
    assert res.final_message == "final summary"
    assert "calls" in res.usage


def test_workflow_route_step() -> None:
    specs = {"fast": _spec("fast")}
    llm = MockLLMClient([
        text_response('{"subagent": "fast", "task_prompt": "X"}'),  # router
        text_response("routed ok"),  # chosen child
    ])
    parent = Agent(llm=llm, permission_policy=PermissivePolicy())
    res = Workflow(parent).route("pick", specs, "do it").run()
    assert res.outputs["pick"] == "routed ok"
    assert res.final_message == "routed ok"


def test_workflow_duplicate_step_rejected() -> None:
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    wf = Workflow(parent).run_agent("a", "hi")
    with pytest.raises(ValueError, match="duplicate step name"):
        wf.run_agent("a", "again")