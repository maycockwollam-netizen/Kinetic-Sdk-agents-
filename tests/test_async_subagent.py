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
    SubagentStatus,
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


async def test_async_agent_auto_upgrades_sync_delegate_tool():
    from kinetic_sdk.subagent.tool import DelegateTool as SyncDelegateTool

    sync_tool = SyncDelegateTool([_spec("worker")])
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([]),
        tools=[sync_tool],
        permission_policy=PermissivePolicy(),
    )
    adopted = parent._tools["delegate"]
    assert isinstance(adopted, AsyncDelegateTool)
    assert adopted is not sync_tool
    assert adopted.parent is parent
    # Async-native: the sync execute raises (never bridged into a thread).
    with pytest.raises(NotImplementedError):
        adopted.execute()
    # add_tool() upgrades too (on a fresh agent, since the name would collide
    # on this one).
    parent2 = AsyncAgent(
        llm=AsyncMockLLMClient([]),
        tools=[],
        permission_policy=PermissivePolicy(),
    )
    parent2.add_tool(SyncDelegateTool([_spec("worker2")]))
    assert isinstance(parent2._tools["delegate"], AsyncDelegateTool)


async def test_async_structured_result_carries_status():
    result_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "needs_more_info", "out_of_scope", "error"],
            },
            "reason": {"type": "string"},
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    spec = SubagentSpec(
        name="structured",
        system_prompt="p",
        description="d",
        result_schema=result_schema,
    )
    parent = AsyncAgent(
        llm=AsyncMockLLMClient(
            [_done('{"status": "out_of_scope", "reason": "not my expertise"}')]
        ),
        permission_policy=PermissivePolicy(),
    )
    result = await run_async_subagent(
        parent, spec, "do it", SpawnBudget(100)
    )
    assert result.status is SubagentStatus.OUT_OF_SCOPE
    assert result.data == {"status": "out_of_scope", "reason": "not my expertise"}


async def test_async_back_pressure_injects_wrap_up():
    from kinetic_sdk.subagent.async_delegation import _GuardedAsyncLLMClient
    from kinetic_sdk.subagent.budget import RepetitionCircuitBreaker
    from kinetic_sdk.subagent.delegation import BACK_PRESSURE_INSTRUCTION

    budget = SpawnBudget(max_total_tool_calls=2)
    budget.record_tool_call("a", "x")  # remaining == 1 -> trigger
    seen: dict[str, list] = {}

    class Recording(AsyncMockLLMClient):
        async def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[override]
            seen["messages"] = list(messages)
            return _done("wrapped up")

    guard = _GuardedAsyncLLMClient(
        Recording([]),
        budget=budget,
        breaker=RepetitionCircuitBreaker(3),
        agent_id="a",
    )
    await guard.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "x", "type": "function", "function": {}}],
    )
    assert any(
        m.get("role") == "user" and m.get("content") == BACK_PRESSURE_INSTRUCTION
        for m in seen["messages"]
    )


async def test_async_factory_called_for_inherited_model():
    # spec.model is None (the common case) — the factory must still build a
    # per-worker client (mirroring the sync spawn), NOT silently share one.
    built: list[str] = []

    def factory(model: str) -> AsyncMockLLMClient:
        built.append(model)
        return AsyncMockLLMClient([_done("ok")])

    parent = AsyncAgent(llm=AsyncMockLLMClient([]), permission_policy=PermissivePolicy())
    spec = SubagentSpec(name="w", system_prompt="p", description="d")
    result = await run_async_subagent(
        parent, spec, "t", SpawnBudget(100), llm_factory=factory
    )
    assert built, "llm_factory was not called for an inherited model"
    assert result.final_message == "ok"


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

async def test_async_root_tool_calls_share_budget_when_opted_in():
    budget = SpawnBudget(1)
    spec = _spec()
    delegate = AsyncDelegateTool([spec], budget=budget, charge_root_tool_calls=True)
    tool = AsyncMockTool(name="noop", result="ok")
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([_call_tool("noop", {}), _call_tool("noop", {})]),
        tools=[delegate, tool],
        permission_policy=PermissivePolicy(),
    )
    delegate.bind(parent)

    with pytest.raises(BudgetExceededError):
        await parent.run("call twice")

    assert budget.used == 1
    assert budget.calls_by_agent() == {"root": 1}


async def test_async_streaming_subagent_charges_budget_and_breaker():
    spec = _spec()
    budget = SpawnBudget()
    parent = AsyncAgent(
        llm=AsyncMockLLMClient([
            _call_tool("noop", {"x": 1}),
            _call_tool("noop", {"x": 1}),
        ]),
        tools=[AsyncMockTool(name="noop", result="ok")],
        permission_policy=PermissivePolicy(),
    )
    child = spawn_async_subagent(parent, spec, budget, max_consecutive_repeats=1)

    with pytest.raises(RepetitionLimitError):
        await child.run("stream", stream=True)

    assert budget.used == 2
