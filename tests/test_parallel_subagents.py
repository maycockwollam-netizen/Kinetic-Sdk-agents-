"""Concurrency coverage for parallel DelegateTool calls.

The parent emits two delegate calls in one response, which sends both child
runs through ``Agent._execute_tool_calls_parallel``. Each child performs two
barrier-synchronised tool calls, making the shared event, audit and
observability sinks receive real concurrent traffic rather than merely
parallel setup work.
"""

from __future__ import annotations

import json
import threading

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.llm.client import LLMResponse, ToolCall
from kinetic_sdk.observability import InMemoryObservabilityLogger
from kinetic_sdk.observability.metrics import MetricsCollector
from kinetic_sdk.security.audit import JSONLAuditLogger
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import DelegateTool, SpawnBudget, SubagentSpec
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from kinetic_sdk.tool.base import Tool, ToolResult


class _SynchronisedWorkTool(Tool):
    """Keep paired child calls concurrent without adding timing flakiness."""

    name = "work"
    description = "Perform one deterministic unit of child work."
    parameters = {
        "type": "object",
        "properties": {"step": {"type": "integer"}},
        "required": ["step"],
    }

    def __init__(self) -> None:
        self._barrier = threading.Barrier(2)

    def execute(self, step: int) -> ToolResult:  # type: ignore[override]
        self._barrier.wait(timeout=5)
        return ToolResult(output={"step": step})


def _delegation_batch(run_number: int) -> LLMResponse:
    """One parent turn that forces Agent's existing parallel tool path."""
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id=f"alpha-{run_number}",
                name="delegate",
                arguments={"subagent_name": "alpha", "task_prompt": "alpha task"},
            ),
            ToolCall(
                id=f"beta-{run_number}",
                name="delegate",
                arguments={"subagent_name": "beta", "task_prompt": "beta task"},
            ),
        ],
        stop_reason="tool_use",
    )


def _child_llm(model: str) -> MockLLMClient:
    """Give every spawned agent an independent, two-tool-call script."""
    return MockLLMClient(
        [
            tool_response(f"{model}-work-1", "work", {"step": 1}),
            tool_response(f"{model}-work-2", "work", {"step": 2}),
            text_response(f"{model} complete"),
        ],
        model=model,
    )


def test_parallel_delegate_calls_keep_shared_sinks_consistent(tmp_path) -> None:
    """Repeated concurrent sub-agent runs leave complete audit and event data."""
    runs = 20
    specs = [
        SubagentSpec(
            name="alpha",
            description="Complete the alpha work.",
            system_prompt="You are alpha.",
            model="alpha-model",
        ),
        SubagentSpec(
            name="beta",
            description="Complete the beta work.",
            system_prompt="You are beta.",
            model="beta-model",
        ),
    ]
    responses = [
        response
        for run_number in range(runs)
        for response in (_delegation_batch(run_number), text_response("parent complete"))
    ]
    budget = SpawnBudget(max_total_tool_calls=runs * 4 + 1)
    delegate = DelegateTool(specs, budget=budget, llm_factory=_child_llm)
    observability = InMemoryObservabilityLogger()
    metrics = MetricsCollector()

    audit_path = tmp_path / "parallel-subagents.jsonl"
    with JSONLAuditLogger(audit_path) as audit:
        parent = Agent(
            llm=MockLLMClient(responses),
            tools=[delegate, _SynchronisedWorkTool()],
            permission_policy=PermissivePolicy(),
            audit_logger=audit,
            observability_logger=observability,
            parallel_tool_execution=True,
        )
        metrics.attach(parent.event_bus)
        delegate.bind(parent)

        for _ in range(runs):
            assert parent.run("delegate both tasks") == "parent complete"

        metrics.detach(parent.event_bus)

    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    # Per parent run: 2 delegate calls/results, 4 child work calls/results,
    # and one spawn + finish record for each child.
    assert len(entries) == runs * 16
    assert budget.used == runs * 4
    calls_by_agent = budget.calls_by_agent()
    assert len(calls_by_agent) == runs * 2
    assert set(calls_by_agent.values()) == {2}

    assert not observability.get_events("agent.error")
    assert not observability.get_events("hooks.error")
    snapshot = metrics.snapshot()
    assert snapshot["tool_calls"] == runs * 6
    assert snapshot["tool_failures"] == 0
