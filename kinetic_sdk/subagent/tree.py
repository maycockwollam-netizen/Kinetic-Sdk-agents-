"""Rebuild the sub-agent delegation tree from a shared audit trail.

Delegation is deliberately decentralised (it is an ordinary tool, no hard
depth cap): there is no central "tree object" to query. Instead, every
spawn and readout is written to the SHARED audit logger as
``subagent_spawn`` / ``subagent_finished`` entries carrying the stable
``parent_agent_id`` / ``agent_id`` pair (see ``delegation.py``). This module
turns that append-only log into a queriable tree again, so a UI, logger or
debugger can show the whole parent -> child hierarchy with each node's
outcome — without the parents having tracked anything centrally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable


@dataclass(frozen=True)
class SubagentNode:
    """One spawned sub-agent as reconstructed from the audit trail.

    Attributes:
        agent_id: The sub-agent's stable delegation id (aka ``agent_id`` in
            ``subagent_spawn`` / ``subagent_finished``).
        spec_name: The ``SubagentSpec.name`` that was spawned.
        parent_agent_id: The spawning agent's id; ``None`` means the parent
            is outside this tree (the outer task that spawned the root).
        outcome: ``completed`` / ``budget_exceeded`` / ``repetition_limit`` /
            ``error``, or ``running`` when no ``subagent_finished`` was seen
            (e.g. the child was spawned but the log is from a crashed run).
        detail: The audit ``detail`` of the finish entry (failure text).
        model: The model the child ran on, when recorded.
        budget_used / budget_max: The tree budget snapshot at spawn time.
    """

    agent_id: str
    spec_name: str
    parent_agent_id: str | None
    outcome: str = "running"
    detail: str | None = None
    model: str | None = None
    budget_used: int | None = None
    budget_max: int | None = None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class DelegationTree:
    """A queriable view of one delegation tree (keyed by agent id)."""

    nodes: dict[str, SubagentNode]

    def __len__(self) -> int:
        return len(self.nodes)

    def root_agent_ids(self) -> list[str]:
        """Agent ids whose parent is outside this tree (the delegation roots)."""
        inside = set(self.nodes)
        return sorted(
            agent_id
            for agent_id, node in self.nodes.items()
            if node.parent_agent_id is None or node.parent_agent_id not in inside
        )

    def children_of(self, agent_id: str) -> list[SubagentNode]:
        """Direct children spawned by *agent_id*, in stable id order."""
        return sorted(
            (node for node in self.nodes.values() if node.parent_agent_id == agent_id),
            key=lambda node: node.agent_id,
        )

    def descendants_of(self, agent_id: str) -> list[SubagentNode]:
        """Every node reachable below *agent_id* (depth-first, stable order).

        Includes the root *agent_id* itself at position 0.
        """
        ordered: list[SubagentNode] = []
        stack = [agent_id]
        while stack:
            current = stack.pop(0)
            node = self.nodes.get(current)
            if node is None:
                continue
            ordered.append(node)
            stack[0:0] = [child.agent_id for child in self.children_of(current)]
        return ordered

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Serialisable mapping (agent_id -> node fields)."""
        return {agent_id: asdict(node) for agent_id, node in self.nodes.items()}


def reconstruct_delegation_tree(entries: Iterable[dict[str, Any]]) -> DelegationTree:
    """Rebuild the delegation tree from audit *entries*.

    *entries* is the audit trail (e.g. ``InMemoryAuditLogger.entries``, or a
    loaded ``JSONLAuditLogger`` file). Only ``subagent_spawn`` and
    ``subagent_finished`` records participate; everything else is ignored.
    A spawned agent with no matching finish record stays ``running``.
    """
    spawns: dict[str, SubagentNode] = {}
    finished: dict[str, tuple[str, str | None]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event = entry.get("event")
        agent_id = entry.get("agent_id")
        if not isinstance(agent_id, str):
            continue
        if event == "subagent_spawn":
            spawns[agent_id] = SubagentNode(
                agent_id=agent_id,
                spec_name=entry.get("spec_name") or "",
                parent_agent_id=entry.get("parent_agent_id"),
                model=entry.get("model"),
                budget_used=_as_int(entry.get("budget_used")),
                budget_max=_as_int(entry.get("budget_max")),
                outcome="running",
            )
        elif event == "subagent_finished":
            finished[agent_id] = (entry.get("outcome") or "error", entry.get("detail"))

    nodes: dict[str, SubagentNode] = {}
    for agent_id, node in spawns.items():
        outcome, detail = finished.get(agent_id, ("running", None))
        nodes[agent_id] = replace(node, outcome=outcome, detail=detail)
    return DelegationTree(nodes)