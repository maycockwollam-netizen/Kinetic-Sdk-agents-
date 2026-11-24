"""Tests for subagent/delegation.py — spawn_subagent + run_subagent."""

from __future__ import annotations

from typing import Any

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    BudgetExceededError,
    DelegateTool,
    RepetitionLimitError,
    SpawnBudget,
    SubagentError,
    SubagentSpec,
    SubagentStatus,
    agent_id_for,
    run_subagent,
    spawn_subagent,
)
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response

from ._helpers import EchoTool, LoopLLM

CHILD_SPEC = SubagentSpec(
    name="child", system_prompt="You are the child.", description="Child agent."
)
GRANDCHILD_SPEC = SubagentSpec(
    name="grandchild",
    system_prompt="You are the grandchild.",
    description="Grandchild agent.",
)


def test_spawn_filters_tools_by_spec_allowlist() -> None:
    from kinetic_sdk.testing import MockTool

    parent = Agent(
        llm=MockLLMClient([]),
        tools=[MockTool(name="alpha"), MockTool(name="beta")],
        permission_policy=PermissivePolicy(),
    )
    spec = SubagentSpec(name="limited", system_prompt="p", tools=("alpha",))
    child = spawn_subagent(parent, spec, SpawnBudget(100))
    assert set(child._tools.keys()) == {"alpha"}  # noqa: SLF001


def test_spawn_inherits_all_tools_when_allowlist_empty() -> None:
    from kinetic_sdk.testing import MockTool

    parent = Agent(
        llm=MockLLMClient([]),
        tools=[MockTool(name="alpha"), MockTool(name="beta")],
        permission_policy=PermissivePolicy(),
    )
    spec = SubagentSpec(name="open", system_prompt="p")
    child = spawn_subagent(parent, spec, SpawnBudget(100))
    assert set(child._tools.keys()) == {"alpha", "beta"}  # noqa: SLF001


def make_parent(
    llm: MockLLMClient | LoopLLM | None = None,
    budget: SpawnBudget | None = None,
    audit: InMemoryAuditLogger | None = None,
) -> tuple[Agent, DelegateTool, EchoTool, InMemoryAuditLogger]:
    """A bound parent agent with one real tool + one DelegateTool."""
    audit = audit if audit is not None else InMemoryAuditLogger()
    delegate = DelegateTool(
        [CHILD_SPEC, GRANDCHILD_SPEC], budget=budget or SpawnBudget(100)
    )
    echo = EchoTool()
    parent = Agent(
        llm=llm or MockLLMClient([]),
        tools=[echo, delegate],
        permission_policy=PermissivePolicy(),
        audit_logger=audit,
    )
    delegate.bind(parent)
    return parent, delegate, echo, audit


class TestSpawnInheritance:
    def test_subagent_inherits_all_parent_tools(self) -> None:
        parent, delegate, echo, _ = make_parent()
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        names = {t["name"] for t in sub.tool_schemas()}
        assert names == {"echo", "delegate"}
        # Non-delegate tools are inherited as the SAME objects...
        assert sub._tools["echo"] is echo  # noqa: SLF001
        # ...while DelegateTool is a clone re-bound to the sub-agent.
        clone = sub._tools["delegate"]  # noqa: SLF001
        assert clone is not delegate
        assert clone.parent is sub
        assert clone.subagent_specs == delegate.subagent_specs

    def test_subagent_has_own_system_prompt_and_fresh_context(self) -> None:
        parent, delegate, _, _ = make_parent()
        parent.state.system_prompt = "parent system prompt"
        parent.state.add_user_message("parent history the child must NOT see")
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        assert sub.state.system_prompt == "You are the child."
        assert sub.state.system_prompt != parent.state.system_prompt
        assert len(sub.state.messages) == 0

    def test_subagent_shares_policy_audit_bus_and_config(self) -> None:
        parent, delegate, _, _ = make_parent()
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        assert sub.permission_policy is parent.permission_policy
        assert sub.audit_logger is parent.audit_logger
        assert sub.event_bus is parent.event_bus
        assert sub.classifier is parent.classifier
        assert sub.context_manager is parent.context_manager
        assert sub.model_context_limit == parent.model_context_limit
        assert sub.hooks is parent.hooks

    def test_budget_instance_is_shared_not_copied(self) -> None:
        parent, delegate, _, _ = make_parent()
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        clone = sub._tools["delegate"]  # noqa: SLF001
        assert clone.budget is delegate.budget

    def test_each_subagent_gets_its_own_circuit_breaker(self) -> None:
        parent, delegate, _, _ = make_parent()
        sub_a = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        sub_b = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        assert sub_a.llm._breaker is not sub_b.llm._breaker  # noqa: SLF001

    def test_no_guard_stacking_on_nested_spawn(self) -> None:
        # A sub-agent's llm is already guarded; the grandchild must wrap the
        # ORIGINAL client, not the parent's guard (else double charging).
        parent, delegate, _, _ = make_parent()
        llm = parent.llm
        child = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        grandchild = spawn_subagent(child, GRANDCHILD_SPEC, delegate.budget)
        assert child.llm.inner is llm
        assert grandchild.llm.inner is llm


class TestModelOverride:
    def test_default_inherits_parent_client(self) -> None:
        parent, delegate, _, _ = make_parent()
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        assert sub.llm.inner is parent.llm
        assert sub.llm.model == parent.llm.model

    def test_same_model_string_needs_no_factory(self) -> None:
        parent, delegate, _, _ = make_parent()
        spec = SubagentSpec(
            name="child", system_prompt="p", model=parent.llm.model
        )
        sub = spawn_subagent(parent, spec, delegate.budget)
        assert sub.llm.inner is parent.llm

    def test_different_model_uses_factory(self) -> None:
        parent, delegate, _, _ = make_parent()
        built: list[str] = []

        def factory(model: str) -> MockLLMClient:
            built.append(model)
            return MockLLMClient([], model=model)

        spec = SubagentSpec(name="child", system_prompt="p", model="cheap-model")
        sub = spawn_subagent(parent, spec, delegate.budget, llm_factory=factory)
        assert built == ["cheap-model"]
        assert sub.llm.inner is not parent.llm
        assert sub.llm.model == "cheap-model"

    def test_different_model_without_factory_raises(self) -> None:
        parent, delegate, _, _ = make_parent()
        spec = SubagentSpec(name="child", system_prompt="p", model="cheap-model")
        with pytest.raises(SubagentError, match="llm_factory"):
            spawn_subagent(parent, spec, delegate.budget)


class TestRunSubagent:
    def test_parent_child_channel_is_task_prompt_only(self) -> None:
        llm = MockLLMClient([text_response("child final answer")])
        parent, delegate, _, _ = make_parent(llm=llm)
        parent.state.add_user_message("secret parent context")
        result = run_subagent(parent, CHILD_SPEC, "do the task", delegate.budget)
        assert result.final_message == "child final answer"
        # The child's only turn: system prompt from the spec, history = just
        # the task_prompt. The parent's prior message never appears.
        (call,) = llm.calls
        assert call["system"] == "You are the child."
        assert call["messages"] == [{"role": "user", "content": "do the task"}]

    def test_nested_spawn_charges_the_same_budget_once_per_call(self) -> None:
        # Script consumed depth-first: child delegates to grandchild, the
        # grandchild calls echo once, both then answer. Total tree tool
        # calls: child's delegate + grandchild's echo = 2.
        llm = MockLLMClient(
            [
                tool_response(
                    "c1", "delegate",
                    {"subagent_name": "grandchild", "task_prompt": "gc task"},
                ),
                tool_response("g1", "echo", {"message": "hi"}),
                text_response("gc done"),
                text_response("child done"),
            ]
        )
        budget = SpawnBudget(100)
        parent, _, _, _ = make_parent(llm=llm, budget=budget)
        result = run_subagent(parent, CHILD_SPEC, "task", budget)
        assert result.final_message == "child done"
        assert budget.used == 2
        # One call charged by the child, one by the grandchild.
        assert sorted(budget.calls_by_agent().values()) == [1, 1]

    def test_budget_exceeded_propagates_out_of_run(self) -> None:
        loop_llm = LoopLLM("echo", {"message": "again"})
        parent, delegate, _, _ = make_parent()
        spec = SubagentSpec(
            name="child", system_prompt="p", model="loop-model"
        )
        budget = SpawnBudget(3)
        with pytest.raises(BudgetExceededError):
            run_subagent(
                parent, spec, "loop", budget,
                llm_factory=lambda model: loop_llm,
            )

    def test_repetition_limit_propagates_out_of_run(self) -> None:
        loop_llm = LoopLLM("echo", {"message": "same every time"})
        parent, delegate, _, _ = make_parent()
        spec = SubagentSpec(name="child", system_prompt="p", model="loop-model")
        with pytest.raises(RepetitionLimitError):
            run_subagent(
                parent, spec, "loop", delegate.budget,
                max_consecutive_repeats=5,
                llm_factory=lambda model: loop_llm,
            )


class TestAuditTrail:
    def test_spawn_and_finish_are_audited_with_tree_ids(self) -> None:
        llm = MockLLMClient([text_response("done")])
        parent, delegate, _, audit = make_parent(llm=llm)
        result = run_subagent(parent, CHILD_SPEC, "task", delegate.budget)
        events = [e for e in audit.entries if e["event"].startswith("subagent")]
        assert [e["event"] for e in events] == ["subagent_spawn", "subagent_finished"]
        spawn, finish = events
        assert spawn["parent_agent_id"] == agent_id_for(parent)
        assert spawn["agent_id"] == result.agent_id
        assert spawn["spec_name"] == "child"
        assert finish["agent_id"] == result.agent_id
        assert finish["outcome"] == "completed"
        assert finish["detail"] == "done"

    def test_parent_id_is_stable_across_spawns(self) -> None:
        parent, delegate, _, audit = make_parent()
        spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        spawn_subagent(parent, GRANDCHILD_SPEC, delegate.budget)
        spawns = [e for e in audit.entries if e["event"] == "subagent_spawn"]
        assert len(spawns) == 2
        assert spawns[0]["parent_agent_id"] == spawns[1]["parent_agent_id"]
        assert spawns[0]["agent_id"] != spawns[1]["agent_id"]

    def test_blocked_run_is_audited_with_outcome(self) -> None:
        loop_llm = LoopLLM("echo", {"message": "again"})
        audit = InMemoryAuditLogger()
        parent, delegate, _, _ = make_parent(audit=audit)
        spec = SubagentSpec(name="child", system_prompt="p", model="loop-model")
        with pytest.raises(BudgetExceededError):
            run_subagent(
                parent, spec, "loop", SpawnBudget(1),
                llm_factory=lambda model: loop_llm,
            )
        finishes = [e for e in audit.entries if e["event"] == "subagent_finished"]
        assert [e["outcome"] for e in finishes] == ["budget_exceeded"]


class TestAgentIds:
    def test_agent_id_for_is_stable_and_unique(self) -> None:
        parent, delegate, _, _ = make_parent()
        sub = spawn_subagent(parent, CHILD_SPEC, delegate.budget)
        assert agent_id_for(parent) == agent_id_for(parent)
        assert agent_id_for(sub) == agent_id_for(sub)
        assert agent_id_for(parent) != agent_id_for(sub)

    def test_two_plain_agents_get_distinct_ids(self) -> None:
        a = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
        b = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
        assert agent_id_for(a) != agent_id_for(b)

class TestRootAndStreamingGuards:
    def test_root_tool_calls_share_budget_when_opted_in(self) -> None:
        budget = SpawnBudget(1)
        delegate = DelegateTool(
            [CHILD_SPEC], budget=budget, charge_root_tool_calls=True
        )
        echo = EchoTool()
        parent = Agent(
            llm=MockLLMClient([
                tool_response("one", "echo", {"message": "first"}),
                tool_response("two", "echo", {"message": "second"}),
            ]),
            tools=[delegate, echo],
            permission_policy=PermissivePolicy(),
        )
        delegate.bind(parent)

        with pytest.raises(BudgetExceededError):
            parent.run("call echo twice")

        assert budget.used == 1
        assert budget.calls_by_agent() == {"root": 1}

    def test_streaming_subagent_charges_budget_and_breaker(self) -> None:
        parent, delegate, _, _ = make_parent(
            llm=MockLLMClient([
                tool_response("one", "echo", {"message": "same"}),
                tool_response("two", "echo", {"message": "same"}),
            ])
        )
        child = spawn_subagent(
            parent, CHILD_SPEC, delegate.budget, max_consecutive_repeats=1
        )

        with pytest.raises(RepetitionLimitError):
            child.run("stream", stream=True)

        assert delegate.budget.used == 2


def test_isolate_tools_rejects_a_tool_that_cannot_be_shared() -> None:
    from kinetic_sdk.tool import Tool, ToolResult

    class MutableTool(Tool):
        name = "mutable"
        description = "Mutable state; neither thread_safe nor cloneable."
        parameters: dict[str, Any] = {"type": "object"}

        def execute(self, **params: Any) -> ToolResult:
            return ToolResult(output="ok")

    parent = Agent(
        llm=MockLLMClient([]),
        tools=[MutableTool()],
        permission_policy=PermissivePolicy(),
    )
    spec = SubagentSpec(name="child", system_prompt="p")
    with pytest.raises(SubagentError, match="cannot be shared"):
        spawn_subagent(parent, spec, SpawnBudget(100), isolate_tools=True)


def test_isolate_tools_clones_cloneable_and_shares_thread_safe() -> None:
    from kinetic_sdk.testing import MockTool
    from kinetic_sdk.tool import Tool, ToolResult

    class CloneableTool(Tool):
        name = "cloneable"
        description = "Implements clone_for_subagent."
        parameters: dict[str, Any] = {"type": "object"}

        def execute(self, **params: Any) -> ToolResult:
            return ToolResult(output="ok")

        def clone_for_subagent(self) -> "CloneableTool":
            return CloneableTool()

    shared = MockTool(name="shared_mock")
    parent = Agent(
        llm=MockLLMClient([]),
        tools=[CloneableTool(), shared],
        permission_policy=PermissivePolicy(),
    )
    child = spawn_subagent(
        parent, SubagentSpec(name="child", system_prompt="p"), SpawnBudget(100), isolate_tools=True
    )

    assert child._tools["cloneable"] is not parent._tools["cloneable"]  # noqa: SLF001
    assert child._tools["shared_mock"] is shared  # noqa: SLF001


def test_subagent_charges_cost_from_usage() -> None:
    from kinetic_sdk.llm.client import LLMResponse

    def charged_llm(model: str) -> MockLLMClient:
        return MockLLMClient(
            [
                LLMResponse(
                    content="child done",
                    usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.012},
                    stop_reason="end_turn",
                )
            ]
        )

    parent, _, _, _ = make_parent()
    budget = SpawnBudget(max_total_cost_usd=0.05)
    result = run_subagent(
        parent,
        SubagentSpec(name="child", system_prompt="p", model="x"),
        "do the task",
        budget,
        llm_factory=charged_llm,
    )

    assert result.final_message == "child done"
    assert budget.used_cost_usd == pytest.approx(0.012)
    assert budget.used_tokens == 15


def test_structured_result_carries_data_and_status() -> None:
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
        system_prompt="You answer with JSON only.",
        description="d",
        result_schema=result_schema,
    )
    child_llm = MockLLMClient(
        [
            text_response(
                '{"status": "needs_more_info", "reason": "need the file paths"}'
            )
        ]
    )
    parent = Agent(llm=child_llm, permission_policy=PermissivePolicy())
    result = run_subagent(
        parent, spec, "figure out what is missing", SpawnBudget(100)
    )

    assert result.status is SubagentStatus.NEEDS_MORE_INFO
    assert result.data == {"status": "needs_more_info", "reason": "need the file paths"}
    assert "need the file paths" in result.final_message


def test_plain_text_handoff_keeps_success_status_when_no_schema() -> None:
    spec = SubagentSpec(name="plain", system_prompt="p", description="d")
    parent = Agent(
        llm=MockLLMClient([text_response("plain done")]),
        permission_policy=PermissivePolicy(),
    )
    result = run_subagent(parent, spec, "go", SpawnBudget(100))
    assert result.status is SubagentStatus.SUCCESS
    assert result.data is None
    assert result.final_message == "plain done"


def test_back_pressure_injects_wrap_up_when_budget_low() -> None:
    from kinetic_sdk.subagent.budget import RepetitionCircuitBreaker, SpawnBudget
    from kinetic_sdk.subagent.delegation import (
        BACK_PRESSURE_INSTRUCTION,
        _GuardedLLMClient,
    )

    budget = SpawnBudget(max_total_tool_calls=2)
    budget.record_tool_call("a", "x")  # remaining == 1 -> trigger back-pressure
    seen: dict[str, list] = {}

    class Recording(MockLLMClient):
        def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[override]
            seen["messages"] = list(messages)
            return text_response("wrapped up")

    guard = _GuardedLLMClient(
        Recording([]), budget=budget, breaker=RepetitionCircuitBreaker(3), agent_id="a"
    )
    guard.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"name": "x", "type": "function", "function": {}}],
    )
    assert any(
        m.get("role") == "user" and m.get("content") == BACK_PRESSURE_INSTRUCTION
        for m in seen["messages"]
    )


def test_back_pressure_not_injected_when_budget_healthy_or_no_tools() -> None:
    from kinetic_sdk.subagent.budget import RepetitionCircuitBreaker, SpawnBudget
    from kinetic_sdk.subagent.delegation import _GuardedLLMClient

    budget = SpawnBudget(max_total_tool_calls=100)  # plenty left
    seen: dict[str, list] = {}

    class Recording(MockLLMClient):
        def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[override]
            seen["messages"] = list(messages)
            return text_response("ok")

    guard = _GuardedLLMClient(
        Recording([]), budget=budget, breaker=RepetitionCircuitBreaker(3), agent_id="a"
    )
    guard.chat([{"role": "user", "content": "hi"}], tools=[{"name": "x"}])
    assert not any("budget remains" in m.get("content", "") for m in seen["messages"])
    # No tools at all -> never nudged.
    guard.chat([{"role": "user", "content": "hi"}], tools=None)
    assert not any("budget remains" in m.get("content", "") for m in seen["messages"])
