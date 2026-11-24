"""Tests for subagent/tree.py — reconstruct_delegation_tree."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    SpawnBudget,
    SubagentSpec,
    reconstruct_delegation_tree,
    run_subagent,
)
from kinetic_sdk.testing import MockLLMClient, text_response


def test_rebuilds_single_delegation_from_real_audit() -> None:
    audit = InMemoryAuditLogger()
    parent = Agent(
        llm=MockLLMClient([text_response("done")]),
        permission_policy=PermissivePolicy(),
        audit_logger=audit,
    )
    spec = SubagentSpec(name="child", system_prompt="p", description="d")
    run_subagent(parent, spec, "do it", SpawnBudget(100))

    tree = reconstruct_delegation_tree(audit.entries)
    assert len(tree) == 1
    node = next(iter(tree.nodes.values()))
    assert node.spec_name == "child"
    assert node.outcome == "completed"
    # The parent's own id is not an audited sub-agent, so the child is a root.
    assert tree.root_agent_ids() == [node.agent_id]


def test_multi_level_tree_with_outcomes() -> None:
    entries = [
        {"event": "subagent_spawn", "agent_id": "a", "spec_name": "alpha",
         "parent_agent_id": None, "model": "m"},
        {"event": "subagent_finished", "agent_id": "a", "spec_name": "alpha",
         "outcome": "completed", "detail": None},
        {"event": "subagent_spawn", "agent_id": "b", "spec_name": "beta",
         "parent_agent_id": "a", "budget_used": 5, "budget_max": 100},
        {"event": "subagent_finished", "agent_id": "b", "spec_name": "beta",
         "outcome": "budget_exceeded", "detail": "cap"},
        {"event": "subagent_spawn", "agent_id": "c", "spec_name": "gamma",
         "parent_agent_id": "b"},
        # c has no finish record -> stays "running"
    ]
    tree = reconstruct_delegation_tree(entries)
    assert len(tree) == 3
    assert tree.root_agent_ids() == ["a"]
    assert [n.agent_id for n in tree.children_of("a")] == ["b"]
    assert [n.agent_id for n in tree.children_of("b")] == ["c"]
    assert tree.nodes["b"].outcome == "budget_exceeded"
    assert tree.nodes["b"].detail == "cap"
    assert tree.nodes["b"].budget_used == 5
    assert tree.nodes["b"].budget_max == 100
    assert tree.nodes["c"].outcome == "running"
    assert [n.agent_id for n in tree.descendants_of("a")] == ["a", "b", "c"]
    assert tree.as_dict()["a"]["spec_name"] == "alpha"
    # non-participating entries are ignored
    with_extra = entries + [{"event": "tool_call", "tool_name": "nothing"}]
    assert len(reconstruct_delegation_tree(with_extra)) == 3