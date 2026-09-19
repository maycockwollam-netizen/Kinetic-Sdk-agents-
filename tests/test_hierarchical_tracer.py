"""Tests for the optional, framework-agnostic hierarchical span tracker."""

from __future__ import annotations

from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.context.manager import ContextManager
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.observability import Tracer
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.tool.base import Tool, ToolFailureCategory, ToolResult
from tests._helpers import MockLLM, text_response, tool_response


class _TransientOnceTool(Tool):
    name = "flaky"
    description = "Fails once with a transient error, then succeeds."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **params: Any) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(error="temporary", failure_category=ToolFailureCategory.TRANSIENT)
        return ToolResult(output="recovered")


class _SecondCheckCompactor(ContextManager):
    """Makes the test's compaction boundary deterministic without loop changes."""

    def __init__(self) -> None:
        self._checks = 0

    def should_compact(self, state: ConversationState, model_context_limit: int) -> bool:
        self._checks += 1
        return self._checks == 2

    def compact(self, state: ConversationState) -> ConversationState:
        return ConversationState(
            system_prompt=state.system_prompt,
            messages=list(state.messages),
            max_messages=state.max_messages,
            metadata=dict(state.metadata),
        )


def test_tracer_builds_real_nested_spans_for_turn_tool_retry_and_compaction():
    """A two-turn run gives retry and compaction their own timed child spans."""
    tracer = Tracer()
    agent = Agent(
        llm=MockLLM([tool_response("c1", "flaky", {}), text_response("done")]),
        tools=[_TransientOnceTool()],
        permission_policy=PermissivePolicy(),
        observability_logger=tracer,
        max_transient_retries=1,
        transient_retry_backoff_seconds=0,
        context_manager=_SecondCheckCompactor(),
    )

    assert agent.run("x" * 100) == "done"
    assert agent.run_id is not None
    spans = tracer.spans_for_run(agent.run_id)
    tree = tracer.as_tree(agent.run_id)

    assert [span.kind for span in spans] == ["run", "turn", "tool_call", "retry", "turn", "compaction"]
    assert tree.kind == "run"
    assert [child.kind for child in tree.children] == ["turn", "turn"]
    first_turn, second_turn = tree.children
    assert [child.kind for child in first_turn.children] == ["tool_call"]
    assert [child.kind for child in first_turn.children[0].children] == ["retry"]
    assert [child.kind for child in second_turn.children] == ["compaction"]
    assert all(span.end_time is not None and span.end_time > span.start_time for span in spans)


def test_tracer_force_closes_every_open_span_on_error_or_cancellation():
    bus = EventBus()
    tracer = Tracer()
    tracer.attach(bus)

    for run_id, terminal_event in (("failed", "agent.error"), ("stopped", "agent.cancelled")):
        bus.publish(Event("agent.run_started", {"run_id": run_id}))
        bus.publish(Event("agent.turn_started", {"run_id": run_id, "iteration": 0}))
        bus.publish(Event("agent.tool_call_started", {"run_id": run_id, "id": "call-1"}))
        bus.publish(Event("agent.tool_retry", {"run_id": run_id, "id": "call-1", "attempt": 1}))
        bus.publish(Event(terminal_event, {"run_id": run_id}))

        spans = tracer.spans_for_run(run_id)
        assert [span.kind for span in spans] == ["run", "turn", "tool_call", "retry"]
        assert all(span.end_time is not None and span.end_time > span.start_time for span in spans)
