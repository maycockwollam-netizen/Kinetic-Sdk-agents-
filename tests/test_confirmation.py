"""Tests for the Confirmation UX: ON_PERMISSION_CHECK hooks confirming or
declining tool calls the policy flagged with ``requires_confirmation=True``.
"""

from __future__ import annotations

from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.hooks import HookContext, HookPoint, HookRegistry, HookResult
from kinetic_sdk.llm.client import LLMResponse, ToolCall
from kinetic_sdk.security import AllowListPolicy
from kinetic_sdk.testing import MockTool
from kinetic_sdk.tool.base import Tool, ToolResult
from tests._helpers import MockLLM, text_response, tool_response


class SpyTool(Tool):
    """Counts executions so tests can prove whether the tool actually ran."""

    name = "spy"
    description = "Records how many times it was executed."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executions = 0

    def execute(self, **params: Any) -> ToolResult:
        self.executions += 1
        return ToolResult(output="executed")


def _make_agent(hooks: HookRegistry | None) -> tuple[Agent, SpyTool, list[Event]]:
    """An agent whose 'spy' tool requires confirmation for target='destroy'."""
    spy = SpyTool()
    policy = AllowListPolicy(
        always_allow=["spy"],
        require_confirmation_patterns={"spy": ["destroy"]},
    )
    llm = MockLLM(
        [tool_response("c1", "spy", {"target": "destroy"}), text_response("end")]
    )
    bus = EventBus()
    denied: list[Event] = []
    bus.subscribe("security.permission_denied", denied.append)
    agent = Agent(
        llm=llm, tools=[spy], permission_policy=policy, event_bus=bus, hooks=hooks
    )
    return agent, spy, denied


def test_no_hooks_configured_denies_as_before():
    agent, spy, denied = _make_agent(hooks=None)

    assert agent.run("go") == "end"
    assert spy.executions == 0
    assert len(denied) == 1
    assert "requires manual confirmation" in denied[0].payload["reason"]


def test_hooks_without_permission_check_hook_denies():
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: None)  # unrelated point
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1


def test_hook_returning_none_denies():
    seen: list[HookContext] = []
    registry = HookRegistry()
    registry.register(HookPoint.ON_PERMISSION_CHECK, lambda ctx: seen.append(ctx))
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1
    # The observer hook still received the full context.
    assert seen[0].tool_name == "spy"
    assert seen[0].tool_input == {"target": "destroy"}
    assert seen[0].permission_decision is not None
    assert seen[0].permission_decision.requires_confirmation is True


def test_hook_confirming_lets_tool_execute():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=True)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    assert agent.run("go") == "end"
    assert spy.executions == 1
    assert denied == []


def test_hook_declining_keeps_denial():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=False)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 0
    assert len(denied) == 1


def test_one_confirming_hook_among_decliners_is_enough():
    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=False)
    )
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=True)
    )
    agent, spy, denied = _make_agent(hooks=registry)

    agent.run("go")

    assert spy.executions == 1
    assert denied == []


def test_raising_hook_falls_back_to_safe_denial():
    bus_errors: list[Event] = []

    def bad(ctx: HookContext) -> HookResult:
        raise RuntimeError("ui crashed")

    registry = HookRegistry()
    registry.register(HookPoint.ON_PERMISSION_CHECK, bad)
    agent, spy, denied = _make_agent(hooks=registry)
    agent.event_bus.subscribe("hooks.error", bus_errors.append)

    agent.run("go")

    # A crashed confirmation UI must never silently allow the call.
    assert spy.executions == 0
    assert len(denied) == 1
    assert len(bus_errors) == 1
    assert bus_errors[0].payload["point"] == "on_permission_check"


def test_declining_hook_does_not_create_checkpoint(tmp_path):
    from kinetic_sdk.replay import CheckpointManager, JsonFileReplayStore

    registry = HookRegistry()
    registry.register(
        HookPoint.ON_PERMISSION_CHECK, lambda ctx: HookResult(should_continue=False)
    )
    store = JsonFileReplayStore(tmp_path / "checkpoint.json")
    agent, spy, denied = _make_agent(hooks=registry)
    agent.checkpoint_manager = CheckpointManager(store)

    assert agent.run("go") == "end"
    assert spy.executions == 0
    assert len(denied) == 1
    assert store.load() is None


def test_unresolved_confirmation_creates_checkpoint_and_can_be_approved(tmp_path):
    import pytest

    from kinetic_sdk.replay import (
        CheckpointManager,
        JsonFileReplayStore,
        PendingConfirmationError,
        resume_from_confirmation,
    )

    store = JsonFileReplayStore(tmp_path / "checkpoint.json")
    manager = CheckpointManager(store)
    agent, spy, _denied = _make_agent(hooks=None)
    agent.checkpoint_manager = manager

    with pytest.raises(PendingConfirmationError) as pending:
        agent.run("go")

    assert spy.executions == 0
    checkpoint_id = pending.value.checkpoint_id
    replay = store.load()
    assert replay is not None
    assert replay.run_id == checkpoint_id
    assert "replay.snapshot" in [step.event_type for step in replay.steps]

    resumed_spy = SpyTool()
    assert resume_from_confirmation(
        checkpoint_id,
        True,
        checkpoint_manager=manager,
        llm=MockLLM([text_response("end")]),
        tools=[resumed_spy],
    ) == "end"
    assert resumed_spy.executions == 1
    replay = store.load()
    assert replay is not None
    assert [step.event_type for step in replay.steps].count("replay.resumed") == 1


def test_observer_only_confirmation_hook_becomes_pending_checkpoint(tmp_path):
    import pytest

    from kinetic_sdk.replay import (
        CheckpointManager,
        JsonFileReplayStore,
        PendingConfirmationError,
    )

    registry = HookRegistry()
    registry.register(HookPoint.ON_PERMISSION_CHECK, lambda ctx: None)
    agent, spy, _denied = _make_agent(hooks=registry)
    agent.checkpoint_manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))

    with pytest.raises(PendingConfirmationError):
        agent.run("go")
    assert spy.executions == 0


def test_unresolved_confirmation_can_be_declined_without_executing_tool(tmp_path):
    import pytest

    from kinetic_sdk.replay import (
        CheckpointManager,
        JsonFileReplayStore,
        PendingConfirmationError,
        resume_from_confirmation,
    )

    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    agent, _spy, _denied = _make_agent(hooks=None)
    agent.checkpoint_manager = manager
    with pytest.raises(PendingConfirmationError) as pending:
        agent.run("go")

    resumed_spy = SpyTool()
    assert resume_from_confirmation(
        pending.value.checkpoint_id,
        False,
        checkpoint_manager=manager,
        llm=MockLLM([text_response("end")]),
        tools=[resumed_spy],
    ) == "end"
    assert resumed_spy.executions == 0


def _tool_result_ids(messages: list[dict[str, Any]]) -> list[str]:
    """Return tool-result ids in their conversation order."""
    return [
        block["tool_use_id"]
        for message in messages
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def test_pending_confirmation_preserves_batch_before_and_after_parallel_gate(tmp_path):
    """A parallel gate pause must checkpoint every unfinalized call exactly once."""
    import pytest

    from kinetic_sdk.replay import (
        CheckpointManager,
        JsonFileReplayStore,
        PendingConfirmationError,
        resume_from_confirmation,
    )

    calls = [
        ToolCall(id="safe-1", name="safe1", arguments={}),
        ToolCall(id="risky", name="risky", arguments={"action": "confirm"}),
        ToolCall(id="safe-2", name="safe2", arguments={}),
    ]
    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    policy = AllowListPolicy(
        always_allow=["safe1", "risky", "safe2"],
        require_confirmation_patterns={"risky": ["confirm"]},
    )
    agent = Agent(
        llm=MockLLM([LLMResponse(content="", tool_calls=calls, stop_reason="tool_use")]),
        tools=[
            MockTool(name="safe1", result="safe-1 done"),
            MockTool(name="risky", result="risky done"),
            MockTool(name="safe2", result="safe-2 done"),
        ],
        permission_policy=policy,
        checkpoint_manager=manager,
        parallel_tool_execution=True,
    )

    with pytest.raises(PendingConfirmationError) as pending:
        agent.run("do three things")

    assert _tool_result_ids(agent.state.messages) == ["safe-1"]
    assert [call.id for call in agent._pending_remaining_calls] == ["risky", "safe-2"]

    resumed_llm = MockLLM([text_response("end")])
    assert resume_from_confirmation(
        pending.value.checkpoint_id,
        True,
        checkpoint_manager=manager,
        llm=resumed_llm,
        tools=[
            MockTool(name="safe1", result="safe-1 done"),
            MockTool(name="risky", result="risky done"),
            MockTool(name="safe2", result="safe-2 done"),
        ],
    ) == "end"
    assert _tool_result_ids(resumed_llm.calls[0]["messages"]) == ["safe-1", "risky", "safe-2"]


def test_pending_confirmation_preserves_complete_suffix_sequentially(tmp_path):
    """The sequential path follows the same complete-suffix checkpoint contract."""
    import pytest

    from kinetic_sdk.replay import (
        CheckpointManager,
        JsonFileReplayStore,
        PendingConfirmationError,
        resume_from_confirmation,
    )

    calls = [
        ToolCall(id="safe-1", name="safe1", arguments={}),
        ToolCall(id="risky", name="risky", arguments={"action": "confirm"}),
        ToolCall(id="safe-2", name="safe2", arguments={}),
    ]
    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    policy = AllowListPolicy(
        always_allow=["safe1", "risky", "safe2"],
        require_confirmation_patterns={"risky": ["confirm"]},
    )
    agent = Agent(
        llm=MockLLM([LLMResponse(content="", tool_calls=calls, stop_reason="tool_use")]),
        tools=[
            MockTool(name="safe1", result="safe-1 done"),
            MockTool(name="risky", result="risky done"),
            MockTool(name="safe2", result="safe-2 done"),
        ],
        permission_policy=policy,
        checkpoint_manager=manager,
    )

    with pytest.raises(PendingConfirmationError):
        agent.run("do three things")

    assert _tool_result_ids(agent.state.messages) == ["safe-1"]
    assert [call.id for call in agent._pending_remaining_calls] == ["risky", "safe-2"]

    resumed_llm = MockLLM([text_response("end")])
    assert resume_from_confirmation(
        agent.run_id or "",
        True,
        checkpoint_manager=manager,
        llm=resumed_llm,
        tools=[
            MockTool(name="safe1", result="safe-1 done"),
            MockTool(name="risky", result="risky done"),
            MockTool(name="safe2", result="safe-2 done"),
        ],
    ) == "end"
    assert _tool_result_ids(resumed_llm.calls[0]["messages"]) == ["safe-1", "risky", "safe-2"]
