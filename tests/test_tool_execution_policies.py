"""Tests for per-tool execution declarations and orchestration policy."""

from __future__ import annotations

import time
from typing import Any

from kinetic_sdk.agent import Agent
from kinetic_sdk.llm import ToolCall
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from kinetic_sdk.tool import (
    Tool,
    ToolCapability,
    ToolExecutionPolicy,
    ToolFailureCategory,
    ToolResult,
    ToolRiskLevel,
)


class DeclaredTool(Tool):
    name = "declared"
    description = "A declared test tool."
    parameters: dict[str, Any] = {"type": "object"}
    risk_level = ToolRiskLevel.READ_ONLY
    capabilities = frozenset({ToolCapability.LOCAL, ToolCapability.FILESYSTEM})

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(output=params)


class FailingTool(Tool):
    name = "failing"
    description = "Always fails."
    parameters: dict[str, Any] = {"type": "object"}

    def __init__(self) -> None:
        self.calls = 0
        self.execution_policy = ToolExecutionPolicy(
            circuit_breaker_threshold=2, circuit_breaker_cooldown_seconds=0.01
        )

    def execute(self, **params: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(error="dependency unavailable")


class DurationTool(Tool):
    name = "duration"
    description = "Returns successful and failing results."
    parameters: dict[str, Any] = {"type": "object", "properties": {"fail": {"type": "boolean"}}}

    def execute(self, *, fail: bool = False) -> ToolResult:
        time.sleep(0.001)
        return ToolResult(error="expected failure") if fail else ToolResult(output="ok")


class LegacyWriteTool(Tool):
    name = "legacy_write"
    description = "Does not declare an idempotency parameter."
    parameters: dict[str, Any] = {"type": "object"}
    risk_level = ToolRiskLevel.WRITE

    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> ToolResult:
        self.calls += 1
        return ToolResult(output="ok")


class IdempotentWriteTool(Tool):
    name = "idempotent_write"
    description = "Accepts idempotency keys."
    parameters: dict[str, Any] = {"type": "object"}
    risk_level = ToolRiskLevel.WRITE
    execution_policy = ToolExecutionPolicy(max_transient_retries=1)

    def __init__(self) -> None:
        self.keys: list[str | None] = []

    def execute(self, *, idempotency_key: str | None = None) -> ToolResult:
        self.keys.append(idempotency_key)
        if len(self.keys) == 1:
            return ToolResult(error="temporary", failure_category=ToolFailureCategory.TRANSIENT)
        return ToolResult(output="ok")


def _agent(tool: Tool, **kwargs: Any) -> Agent:
    return Agent(
        MockLLMClient([]), tools=[tool], permission_policy=PermissivePolicy(), **kwargs
    )


def test_tool_declarations_are_optional_and_readable() -> None:
    declared = DeclaredTool()
    legacy = LegacyWriteTool()

    assert declared.risk_level is ToolRiskLevel.READ_ONLY
    assert declared.capabilities == frozenset({ToolCapability.LOCAL, ToolCapability.FILESYSTEM})
    assert legacy.risk_level is ToolRiskLevel.WRITE
    assert Tool.risk_level is None
    assert Tool.capabilities == frozenset()
    assert Tool.execution_policy is None


def test_circuit_breaker_opens_per_tool_and_closes_after_cooldown() -> None:
    tool = FailingTool()
    agent = _agent(tool, max_transient_retries=0)

    assert agent._execute_one(ToolCall("first", "failing", {})).is_error
    assert agent._execute_one(ToolCall("second", "failing", {})).is_error
    opened = agent._execute_one(ToolCall("third", "failing", {}))
    assert tool.calls == 2
    assert opened.failure_category is ToolFailureCategory.PERMANENT
    assert "Circuit breaker open" in opened.error

    time.sleep(0.02)
    assert agent._execute_one(ToolCall("after-cooldown", "failing", {})).is_error
    assert tool.calls == 3


def test_agent_measures_duration_for_success_and_failure() -> None:
    agent = _agent(DurationTool())

    successful = agent._execute_one(ToolCall("success", "duration", {"fail": False}))
    failed = agent._execute_one(ToolCall("failure", "duration", {"fail": True}))

    assert successful.duration_seconds is not None and successful.duration_seconds > 0
    assert failed.duration_seconds is not None and failed.duration_seconds > 0


def test_per_tool_timeout_overrides_the_agent_wide_timeout() -> None:
    class SlowTool(DurationTool):
        execution_policy = ToolExecutionPolicy(timeout_seconds=0.001)

        def execute(self, *, fail: bool = False) -> ToolResult:
            time.sleep(0.01)
            return ToolResult(output="too late")

    result = _agent(SlowTool(), tool_timeout=1)._execute_one(
        ToolCall("timeout", "duration", {})
    )

    assert result.is_error is True
    assert "timed out after 0.001s" in result.error
    assert result.duration_seconds is not None and result.duration_seconds > 0


def test_idempotency_key_is_stable_for_write_retries_and_legacy_tools_work() -> None:
    tool = IdempotentWriteTool()
    llm = MockLLMClient(
        [tool_response(name=tool.name, arguments={}), text_response("done")]
    )
    agent = Agent(
        llm,
        tools=[tool],
        permission_policy=PermissivePolicy(),
        max_transient_retries=0,
        transient_retry_backoff_seconds=0,
    )

    assert agent.run("write") == "done"
    assert len(tool.keys) == 2
    assert tool.keys[0] is not None
    assert tool.keys[0] == tool.keys[1]

    legacy = LegacyWriteTool()
    assert _agent(legacy)._execute_one(ToolCall("legacy", legacy.name, {})).output == "ok"
    assert legacy.calls == 1
