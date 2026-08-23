"""End-to-end runaway containment tests for the subagent package.

These simulate the failure mode the whole module exists for: a sub-agent
whose model NEVER stops delegating (infinite recursion) or NEVER stops
repeating one tool call (stuck loop). The guardrails must terminate the
tree quickly and cheaply — long before any timeout.
"""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    DelegateTool,
    SpawnBudget,
    SubagentSpec,
)

from ._helpers import LoopLLM

LOOP_SPEC = SubagentSpec(
    name="loop",
    system_prompt="You must keep delegating the same task forever.",
    description="Runaway fixture: delegates without any stop condition.",
)


def test_runaway_self_delegation_is_contained_by_budget() -> None:
    # The root's model and every inherited sub-agent model ALWAYS answer
    # with another delegate call — an unbounded recursion the depth-agnostic
    # design intentionally allows... until the shared budget runs out.
    budget = SpawnBudget(max_total_tool_calls=30)
    loop_llm = LoopLLM("delegate", {"subagent_name": "loop", "task_prompt": "again"})
    delegate = DelegateTool([LOOP_SPEC], budget=budget)
    audit = InMemoryAuditLogger()
    root = Agent(
        llm=loop_llm,
        tools=[delegate],
        permission_policy=PermissivePolicy(),
        audit_logger=audit,
        max_iterations=5,  # inherited by every spawned sub-agent
    )
    delegate.bind(root)

    final = root.run("start the runaway")

    # Terminated (did not hang), budget fully consumed, never exceeded.
    assert isinstance(final, str)
    assert budget.used == 30
    assert budget.exhausted
    # Every agent in the tree shares the one LoopLLM instance, so this is
    # the TOTAL number of model turns the runaway burned: bounded and
    # cheap (each level stops retrying quickly once the budget is gone).
    assert loop_llm.calls <= 200
    # The audit trail records both the spawns and the throttled outcomes.
    spawns = [e for e in audit.entries if e["event"] == "subagent_spawn"]
    blocked = [
        e
        for e in audit.entries
        if e["event"] == "subagent_finished" and e["outcome"] == "budget_exceeded"
    ]
    # Spawn count is bounded: at most one spawn per charged call (30) plus
    # the root's own uncharged retries (<= its max_iterations of 5).
    assert len(spawns) <= 30 + 5
    assert len(blocked) >= 1


def test_stuck_repeating_subagent_is_contained_by_circuit_breaker() -> None:
    # The child calls echo with identical arguments forever; its OWN
    # breaker (not shared with the parent) trips after 3 consecutive
    # repeats, the delegation returns an error result, and the root
    # finishes normally instead of hanging.
    budget = SpawnBudget(max_total_tool_calls=1000)
    child_llm = LoopLLM("echo", {"message": "stuck"})
    delegate = DelegateTool(
        [
            SubagentSpec(
                name="loop",
                system_prompt="Repeat the echo call forever.",
                description="Stuck fixture.",
                model="loop-model",
            )
        ],
        budget=budget,
        max_consecutive_repeats=3,
        llm_factory=lambda model: child_llm,
    )

    from kinetic_sdk.testing import MockLLMClient, text_response, tool_response

    root_llm = MockLLMClient(
        [
            tool_response(
                "d1", "delegate", {"subagent_name": "loop", "task_prompt": "go"}
            ),
            text_response("root finished after blocked delegation"),
        ]
    )
    audit = InMemoryAuditLogger()
    root = Agent(
        llm=root_llm,
        tools=[delegate],
        permission_policy=PermissivePolicy(),
        audit_logger=audit,
    )
    delegate.bind(root)

    assert root.run("delegate to the stuck agent") == (
        "root finished after blocked delegation"
    )
    # Breaker tripped on the 4th identical call (3 allowed + 1 over), far
    # below the 1000-call budget. The tripping call is still charged to the
    # budget (the model turn already happened), so used == 4.
    assert child_llm.calls == 4
    assert budget.used == 4
    finishes = [e for e in audit.entries if e["event"] == "subagent_finished"]
    assert [e["outcome"] for e in finishes] == ["repetition_limit"]
