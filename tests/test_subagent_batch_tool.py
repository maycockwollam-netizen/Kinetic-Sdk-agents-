"""Tests for DelegateBatchTool — parallel batch delegation."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import SpawnBudget, SubagentSpec
from kinetic_sdk.subagent.batch_tool import DelegateBatchTool
from kinetic_sdk.testing import MockLLMClient, text_response

ALPHA = SubagentSpec(name="alpha", system_prompt="You do alpha work.", description="Alpha.")
BETA = SubagentSpec(name="beta", system_prompt="You do beta work.", description="Beta.")


def _bound_batch(*, specs, max_concurrency=2):
    batch = DelegateBatchTool(
        specs, budget=SpawnBudget(100), max_concurrency=max_concurrency
    )
    parent = Agent(
        llm=MockLLMClient(
            [text_response("result one"), text_response("result two")], model="openai/gpt-4o"
        ),
        tools=[batch],
        permission_policy=PermissivePolicy(),
    )
    batch.bind(parent)
    return batch, parent


def test_delegate_batch_runs_tasks_and_merges() -> None:
    batch, _parent = _bound_batch(specs=[ALPHA, BETA])
    result = batch.execute(
        tasks=[
            {"task": "do one", "subagent_name": "alpha"},
            {"task": "do two", "subagent_name": "beta"},
        ]
    )
    assert not result.is_error, result.error
    assert "result one" in result.output
    assert "result two" in result.output
    assert result.metadata["count"] == 2
    assert result.metadata["subagents"] == ["alpha", "beta"]


def test_delegate_batch_single_spec_defaults_to_it() -> None:
    batch, _parent = _bound_batch(specs=[ALPHA])
    result = batch.execute(tasks=[{"task": "do the thing"}])
    assert not result.is_error, result.error
    assert "result one" in result.output


def test_delegate_batch_requires_subagent_name_when_multiple() -> None:
    batch, _parent = _bound_batch(specs=[ALPHA, BETA])
    result = batch.execute(tasks=[{"task": "do one"}])
    assert result.is_error
    assert "subagent_name" in str(result.error)


def test_delegate_batch_rejects_unknown_subagent() -> None:
    batch, _parent = _bound_batch(specs=[ALPHA])
    result = batch.execute(tasks=[{"task": "x", "subagent_name": "nope"}])
    assert result.is_error
    assert "unknown sub-agent" in str(result.error)


def test_delegate_batch_rejects_empty_tasks() -> None:
    batch, _parent = _bound_batch(specs=[ALPHA])
    for bad in (None, [], [{"task": "   "}], "not-a-list"):
        result = batch.execute(tasks=bad)  # type: ignore[arg-type]
        assert result.is_error


def test_delegate_batch_unbound_tool_errors() -> None:
    batch = DelegateBatchTool([ALPHA], budget=SpawnBudget(100))
    result = batch.execute(tasks=[{"task": "x"}])
    assert result.is_error
    assert "not bound" in str(result.error)