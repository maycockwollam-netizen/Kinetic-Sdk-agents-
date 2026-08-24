"""Async sub-agent delegation: spawn, guardrails, tree budget sharing."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.llm.client import LLMResponse, ToolCall
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    AsyncDelegateTool,
    AsyncLLMFactory,
    BudgetExceededError,
    RepetitionLimitError,
    SpawnBudget,
    SubagentError,
    SubagentSpec,
    async_agent_id_for,
    run_async_subagent,
    spawn_async_subagent,
)
from kinetic_sdk.testing.async_mocks import AsyncMockLLMClient, AsyncMockTool

pytestmark = pytest.mark.asyncio


def _spec(name="worker", model=None):
    return SubagentSpec(
        name=name,
        system_prompt=f"You are {name}.",
        description=f"{name} description",
        model=model,
    )


def _done(text="done"):
    return LLMResponse(content=text, stop_reason="end_turn")


def _call_tool(name, arguments, call_id="c1"):
    return LLMResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


async def test_async_spawn_and_run_inherits_and_completes():
    spec = _spec()
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([]),
        permission_policy=PermissivePolicy(),
    )
    sub_agent = spawn_async_subagent(parent, spec, SpawnBudget())
    assert isinstance(sub_agent, AsyncAgent)
    assert sub_agent.permission_policy is parent.permission_policy
    # Fresh context: spec system prompt, no parent history.
    assert sub_agent.state.system_prompt == "You are worker."
    # Child inherits the parent's client (empty script -> immediate end_turn).
    result = await sub_agent.run("task")
    assert result == ""


async def test_run_async_subagent_roundtrip():
    spec = _spec()
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("root")]),
        permission_policy=PermissivePolicy(),
    )
    budget = SpawnBudget()
    result = await run_async_subagent(parent, spec, "do it", budget)
    assert result.agent_id
    # Audit: spawn + finished logged.
    kinds = [e["event"] for e in parent.audit_logger.entries]
    assert "subagent_spawn" in kinds and "subagent_finished" in kinds


async def test_async_budget_enforced_in_chat():
    spec = _spec()
    # Child inherits the parent's client: two tool-requesting turns with a
    # budget of 1 -> the SECOND charge raises out of the child's run.
    tool = AsyncMockTool(name="noop", result="ok")
    parent_llm = AsyncMockLLMClient(
        [_call_tool("noop", {}), _call_tool("noop", {})], model="parent-model"
    )
    parent = AsyncAgent(
        llm=parent_llm,
        tools=[tool],
        permission_policy=PermissivePolicy(),
    )
    budget = SpawnBudget(max_total_tool_calls=1)
    with pytest.raises(BudgetExceededError):
        await run_async_subagent(parent, spec, "task", budget)


async def test_async_repetition_breaker_trips():
    spec = _spec()
    # Two identical tool-call turns in a row, breaker limit 1 -> second
    # identical request raises.
    responses = [_call_tool("noop", {"x": 1}), _call_tool("noop", {"x": 1})]
    parent = AsyncAgent(
        llm=AsyncMockLLMClient(responses),
        tools=[AsyncMockTool(name="noop", result="ok")],
        permission_policy=PermissivePolicy(),
    )
    with pytest.raises(RepetitionLimitError):
        await run_async_subagent(
            parent, spec, "task", SpawnBudget(), max_consecutive_repeats=1
        )


async def test_spawn_never_stacks_guards():
    spec = _spec()
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("x")]),
        permission_policy=PermissivePolicy(),
    )
    budget = SpawnBudget()
    child = spawn_async_subagent(parent, spec, budget)
    grandchild = spawn_async_subagent(child, _spec("leaf"), budget)
    # The grandchild's client guard wraps the UNGUARDED base client.
    from kinetic_sdk.subagent.async_delegation import _GuardedAsyncLLMClient

    assert isinstance(grandchild.llm, _GuardedAsyncLLMClient)
    assert not isinstance(grandchild.llm.inner, _GuardedAsyncLLMClient)


async def test_spec_model_without_factory_raises():
    spec = _spec(model="other-model")
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("x")], model="parent-model"),
        permission_policy=PermissivePolicy(),
    )
    with pytest.raises(SubagentError):
        spawn_async_subagent(parent, spec, SpawnBudget())


async def test_spec_model_with_factory_used():
    spec = _spec(model="other-model")
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("x")], model="parent-model"),
        permission_policy=PermissivePolicy(),
    )
    factory: AsyncLLMFactory = lambda model: AsyncMockLLMClient(
        [_done("built")], model=model
    )
    sub_agent = spawn_async_subagent(
        parent, spec, SpawnBudget(), llm_factory=factory
    )
    assert sub_agent.llm.model == "other-model"


async def test_async_delegate_tool_end_to_end():
    # The child gets its OWN client via a factory (spec.model differs);
    # otherwise the shared parent client/script is consumed by the child.
    spec = _spec(model="child-model")
    budget = SpawnBudget()
    factory: AsyncLLMFactory = lambda model: AsyncMockLLMClient(
        [_done("child final")], model=model
    )
    tool = AsyncDelegateTool([spec], budget=budget, llm_factory=factory)
    parent = AsyncAgent(
        llm=AsyncMockLLMClient(
            [
                _call_tool("delegate", {"subagent_name": "worker", "task_prompt": "t"}),
                _done("final"),
            ]
        ),
        tools=[tool],
        permission_policy=PermissivePolicy(),
    )
    tool.bind(parent)
    result = await parent.run("delegate the task")
    assert result == "final"
    # The delegation itself passed through the permission policy + audit.
    assert any(e["event"] == "tool_call" for e in parent.audit_logger.entries)


async def test_async_delegate_tool_unbound_and_unknown():
    spec = _spec()
    tool = AsyncDelegateTool([spec])
    result = await tool.execute_async(subagent_name="worker", task_prompt="t")
    assert result.is_error and "not bound" in result.error
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("x")]),
        tools=[tool],
        permission_policy=PermissivePolicy(),
    )
    tool.bind(parent)
    result = await tool.execute_async(subagent_name="nope", task_prompt="t")
    assert result.is_error and "Unknown sub-agent" in result.error


async def test_async_delegate_tool_sync_execute_raises():
    tool = AsyncDelegateTool([_spec()])
    with pytest.raises(NotImplementedError):
        tool.execute(subagent_name="worker", task_prompt="t")


async def test_async_agent_id_stable():
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_done("x")]),
        permission_policy=PermissivePolicy(),
    )
    assert async_agent_id_for(parent) == async_agent_id_for(parent)
