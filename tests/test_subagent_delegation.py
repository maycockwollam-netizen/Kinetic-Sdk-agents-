"""Tests for subagent/delegation.py — spawn_subagent + run_subagent."""

from __future__ import annotations

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
