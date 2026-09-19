"""Confirmation checkpoint coverage for sync and async multi-call batches."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.llm.client import LLMResponse, ToolCall
from kinetic_sdk.replay import (
    CheckpointManager,
    JsonFileReplayStore,
    PendingConfirmationError,
    resume_async_from_confirmation,
    resume_from_confirmation,
)
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.testing import (
    AsyncMockLLMClient,
    AsyncMockTool,
    MockLLMClient,
    MockTool,
    text_response,
)


def _calls() -> list[ToolCall]:
    return [
        ToolCall("c1", "safe1", {}),
        ToolCall("c2", "confirm", {"action": "confirm"}),
        ToolCall("c3", "safe2", {}),
    ]


def _policy() -> AllowListPolicy:
    return AllowListPolicy(
        always_allow=["safe1", "confirm", "safe2"],
        require_confirmation_patterns={"confirm": ["confirm"]},
    )


def _tool_results(messages: list[dict]) -> dict[str, dict]:
    return {
        block["tool_use_id"]: block
        for message in messages
        if message.get("role") == "user"
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("approved", [False, True])
async def test_async_confirmation_batch_has_complete_history_and_does_not_rerun_safe_call(
    tmp_path, parallel: bool, approved: bool
) -> None:
    """Both async paths preserve a provider-valid batch across a pause."""
    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    first = AsyncMockTool("safe1", result="safe one")
    agent = AsyncAgent(
        AsyncMockLLMClient([LLMResponse(tool_calls=_calls(), stop_reason="tool_use")]),
        tools=[first, AsyncMockTool("confirm", result="confirmed"), AsyncMockTool("safe2", result="safe two")],
        permission_policy=_policy(),
        checkpoint_manager=manager,
        parallel_tool_execution=parallel,
    )

    with pytest.raises(PendingConfirmationError) as pending:
        await agent.run("run batch")

    confirm = AsyncMockTool("confirm", result="confirmed")
    last = AsyncMockTool("safe2", result="safe two")
    resumed_llm = AsyncMockLLMClient([text_response("final answer")])
    assert await resume_async_from_confirmation(
        pending.value.checkpoint_id,
        approved,
        checkpoint_manager=manager,
        llm=resumed_llm,
        tools=[first, confirm, last],
        permission_policy=_policy(),
        parallel_tool_execution=parallel,
    ) == "final answer"

    results = _tool_results(resumed_llm.calls[0]["messages"])
    assert set(results) == {"c1", "c2", "c3"}
    assert len(first.calls) == 1
    assert len(confirm.calls) == (1 if approved else 0)
    assert len(last.calls) == 1


@pytest.mark.parametrize("pass_policy", [False, True])
def test_sync_resume_helper_forwards_agent_kwargs_to_remaining_calls(tmp_path, pass_policy: bool) -> None:
    """The sync helper must retain its historical API and forward new kwargs."""
    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    agent = Agent(
        MockLLMClient([LLMResponse(tool_calls=_calls(), stop_reason="tool_use")]),
        tools=[MockTool("safe1", result="safe one"), MockTool("confirm", result="confirmed"), MockTool("safe2", result="safe two")],
        permission_policy=_policy(),
        checkpoint_manager=manager,
    )
    with pytest.raises(PendingConfirmationError) as pending:
        agent.run("run batch")

    last = MockTool("safe2", result="safe two")
    resumed_llm = MockLLMClient([text_response("final answer")])
    kwargs = {"permission_policy": _policy()} if pass_policy else {}
    assert resume_from_confirmation(
        pending.value.checkpoint_id,
        True,
        checkpoint_manager=manager,
        llm=resumed_llm,
        tools=[MockTool("safe1", result="safe one"), MockTool("confirm", result="confirmed"), last],
        **kwargs,
    ) == "final answer"
    results = _tool_results(resumed_llm.calls[0]["messages"])
    assert set(results) == {"c1", "c2", "c3"}
    assert len(last.calls) == (1 if pass_policy else 0)
    assert results["c3"].get("is_error", False) is (not pass_policy)


@pytest.mark.asyncio
@pytest.mark.parametrize("pass_policy", [False, True])
async def test_async_resume_helper_forwards_agent_kwargs_to_remaining_calls(
    tmp_path, pass_policy: bool
) -> None:
    """The async helper forwards policy settings needed by remaining calls."""
    manager = CheckpointManager(JsonFileReplayStore(tmp_path / "checkpoint.json"))
    agent = AsyncAgent(
        AsyncMockLLMClient([LLMResponse(tool_calls=_calls(), stop_reason="tool_use")]),
        tools=[AsyncMockTool("safe1", result="safe one"), AsyncMockTool("confirm", result="confirmed"), AsyncMockTool("safe2", result="safe two")],
        permission_policy=_policy(),
        checkpoint_manager=manager,
    )
    with pytest.raises(PendingConfirmationError) as pending:
        await agent.run("run batch")

    last = AsyncMockTool("safe2", result="safe two")
    resumed_llm = AsyncMockLLMClient([text_response("final answer")])
    kwargs = {"permission_policy": _policy()} if pass_policy else {}
    assert await resume_async_from_confirmation(
        pending.value.checkpoint_id,
        True,
        checkpoint_manager=manager,
        llm=resumed_llm,
        tools=[AsyncMockTool("safe1", result="safe one"), AsyncMockTool("confirm", result="confirmed"), last],
        **kwargs,
    ) == "final answer"
    results = _tool_results(resumed_llm.calls[0]["messages"])
    assert set(results) == {"c1", "c2", "c3"}
    assert len(last.calls) == (1 if pass_policy else 0)
    assert results["c3"].get("is_error", False) is (not pass_policy)
