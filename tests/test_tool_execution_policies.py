"""Tests for per-tool execution declarations and orchestration policy."""

from __future__ import annotations

import time
from typing import Any

import pytest

from kinetic_sdk.agent import Agent
from kinetic_sdk.llm import ToolCall
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response
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


class NonThreadSafeTool(Tool):
    name = "nonts"
    description = "No thread_safe declaration."
    parameters: dict[str, Any] = {"type": "object"}

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(output="ok")


def _multi_call_response(*names: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"c{i}", name=name, arguments={}) for i, name in enumerate(names)],
        stop_reason="tool_use",
    )


def test_parallel_warns_for_a_non_thread_safe_tool() -> None:
    llm = MockLLMClient([_multi_call_response("nonts", "nonts"), text_response("done")])
    agent = Agent(
        llm, tools=[NonThreadSafeTool()], permission_policy=PermissivePolicy(), parallel_tool_execution=True
    )
    with pytest.warns(RuntimeWarning, match="thread_safe=False"):
        assert agent.run("go") == "done"


def test_parallel_thread_safe_violation_is_a_hard_error(monkeypatch) -> None:
    monkeypatch.setenv("KINETIC_ENFORCE_THREAD_SAFE", "error")
    llm = MockLLMClient([_multi_call_response("nonts", "nonts"), text_response("done")])
    agent = Agent(
        llm, tools=[NonThreadSafeTool()], permission_policy=PermissivePolicy(), parallel_tool_execution=True
    )
    with pytest.raises(RuntimeError, match="thread_safe=False"):
        agent.run("go")


def test_parallel_capacity_exhaustion_errors_overflow_calls() -> None:
    import threading

    tool = MockTool(name="echo_mock")
    llm = MockLLMClient([_multi_call_response("echo_mock", "echo_mock"), text_response("done")])
    agent = Agent(
        llm, tools=[tool], permission_policy=PermissivePolicy(), parallel_tool_execution=True
    )
    agent._tool_slots = threading.BoundedSemaphore(0)  # noqa: SLF001
    assert agent.run("go") == "done"
    assert tool.calls == []
    contents = [
        agent.state.messages[index]["content"][0]["content"] for index in (2, 3)
    ]
    assert all("capacity exhausted" in content for content in contents)


def test_agent_does_not_mutate_a_shared_result_instance() -> None:
    shared = ToolResult(output="shared")

    class SharedTool(Tool):
        name = "shared"
        description = "Returns the same ToolResult instance every call."
        parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        def execute(self, **params: Any) -> ToolResult:
            return shared

    agent = _agent(SharedTool())
    first = agent._invoke_tool(agent._tools["shared"], {})  # noqa: SLF001
    second = agent._invoke_tool(agent._tools["shared"], {})  # noqa: SLF001

    assert shared.duration_seconds is None
    assert first is not shared and second is not shared
    assert first.output == "shared" and second.output == "shared"
    assert first.duration_seconds is not None and second.duration_seconds is not None


def test_filetool_write_is_idempotent_under_the_same_key(tmp_path) -> None:
    from kinetic_sdk.files import FileTool
    from kinetic_sdk.workspace import LocalWorkspace

    tool = FileTool(LocalWorkspace(tmp_path))
    # The key is derived from (tool, arguments, run); retrying the same call
    # with the same key must be a no-op that returns the recorded result.
    applied = tool.execute(
        action="str_replace",
        path="a.txt",
        old_str="old",
        new_str="new",
        idempotency_key="k1",
    )
    # First call fails (no file yet) and is NOT cached.
    assert applied.is_error is True
    assert tool.execute(action="create", path="a.txt", file_text="old\n").is_error is False

    applied = tool.execute(
        action="str_replace", path="a.txt", old_str="old", new_str="new", idempotency_key="k1"
    )
    cached = tool.execute(
        action="str_replace", path="a.txt", old_str="old", new_str="new", idempotency_key="k1"
    )
    assert applied.is_error is False
    assert cached.is_error is False
    assert (tmp_path / "a.txt").read_text() == "new\n"
    # A different key applied to now-stale content must fail: the dedup above
    # actually skipped a second apply, it did not make file editing magical.
    stale = tool.execute(
        action="str_replace", path="a.txt", old_str="old", new_str="bye", idempotency_key="k2"
    )
    assert stale.is_error is True


def test_builtin_tools_declare_the_execution_contract() -> None:
    import inspect

    from kinetic_sdk.files import FileTool
    from kinetic_sdk.files.patch import ApplyPatchTool
    from kinetic_sdk.git import GitTool
    from kinetic_sdk.terminal import TerminalTool
    from kinetic_sdk.todo import TodoWriteTool

    for tool_cls in (FileTool, GitTool, TodoWriteTool, ApplyPatchTool):
        assert tool_cls.risk_level is ToolRiskLevel.WRITE
        assert "idempotency_key" in inspect.signature(tool_cls.execute).parameters

    # Stateless per-call built-ins are safe to share across concurrent runs.
    assert TerminalTool.thread_safe is True
    assert GitTool.thread_safe is True
    # FileTool keeps mutable per-instance state (undo + idempotency cache), so
    # it must be cloned rather than shared.
    assert FileTool.thread_safe is False
    assert FileTool.clone_for_subagent is not Tool.clone_for_subagent


def test_submit_timed_tool_releases_slot_when_submit_fails() -> None:
    import threading

    class BoomExecutor:
        def submit(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

    tool = MockTool()
    agent = _agent(tool)
    agent._tool_slots = threading.BoundedSemaphore(1)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="boom"):
        agent._submit_timed_tool(BoomExecutor(), tool, {})  # type: ignore[arg-type]  # noqa: SLF001
    # The slot must be released even though submit failed, so capacity is whole.
    assert agent._tool_slots.acquire(blocking=False)  # noqa: SLF001
