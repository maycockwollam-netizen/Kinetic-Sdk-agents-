"""Recovery and unproductive-loop behaviour in the synchronous agent loop."""

from __future__ import annotations

from kinetic_sdk.agent import Agent, AgentMode, StuckDetector
from kinetic_sdk.agent.classifier import Classification, TaskClassifier, TaskComplexity
from kinetic_sdk.event import EventBus
from kinetic_sdk.event.bus import Event
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response
from kinetic_sdk.tool import ToolFailureCategory, ToolResult


class FlashClassifier(TaskClassifier):
    def classify(self, task: str) -> Classification:
        return Classification(
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            rationale="test",
            mode=AgentMode.FLASH,
        )


def test_transient_tool_failures_are_retried_until_success() -> None:
    outcomes = iter(
        [
            ToolResult(error="temporary outage", failure_category=ToolFailureCategory.TRANSIENT),
            ToolResult(error="temporary outage", failure_category=ToolFailureCategory.TRANSIENT),
            ToolResult(output="recovered"),
        ]
    )
    tool = MockTool("network", handler=lambda **_: next(outcomes))
    llm = MockLLMClient([tool_response(name="network", arguments={}), text_response("done")])
    agent = Agent(
        llm,
        tools=[tool],
        classifier=FlashClassifier(),
        permission_policy=PermissivePolicy(),
        max_transient_retries=2,
        transient_retry_backoff_seconds=0,
    )

    assert agent.run("retry") == "done"
    assert len(tool.calls) == 3
    assert agent.mode is AgentMode.FLASH


def test_permanent_tool_failure_is_not_retried_and_escalates_flash() -> None:
    tool = MockTool(
        "broken",
        result=ToolResult(error="invalid operation", failure_category=ToolFailureCategory.PERMANENT),
    )
    llm = MockLLMClient([tool_response(name="broken", arguments={}), text_response("handled")])
    agent = Agent(
        llm,
        tools=[tool],
        classifier=FlashClassifier(),
        permission_policy=PermissivePolicy(),
        max_transient_retries=2,
        transient_retry_backoff_seconds=0,
    )

    assert agent.run("run") == "handled"
    assert len(tool.calls) == 1
    assert agent.mode is AgentMode.MAX


def test_unclassified_tool_failure_preserves_no_retry_behavior() -> None:
    tool = MockTool("legacy", result=ToolResult(error="legacy failure"))
    llm = MockLLMClient([tool_response(name="legacy", arguments={}), text_response("handled")])
    agent = Agent(
        llm,
        tools=[tool],
        permission_policy=PermissivePolicy(),
        max_transient_retries=2,
        transient_retry_backoff_seconds=0,
    )

    assert agent.run("run") == "handled"
    assert len(tool.calls) == 1


def test_unproductive_loop_stops_before_iteration_cap() -> None:
    bus = EventBus()
    errors: list[Event] = []
    bus.subscribe("agent.error", errors.append)
    loop_call = tool_response(name="echo", arguments={"value": "same"})
    llm = MockLLMClient([loop_call] * 10)
    agent = Agent(
        llm,
        tools=[MockTool("echo", result="unchanged")],
        event_bus=bus,
        permission_policy=PermissivePolicy(),
        max_iterations=10,
        stuck_detector=StuckDetector(window_size=3, repeat_threshold=3),
    )

    assert agent.run("loop") == ""
    assert len(llm.calls) == 3
    assert any(event.payload["reason"] == "unproductive_loop" for event in errors)
