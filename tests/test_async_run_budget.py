"""AsyncAgent integration tests for the root run budget (``agent/budget.py``).

Mirrors the sync ``Agent`` contract: a looping conversation is stopped
gracefully (event + explanatory final text), never crashed.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent import AsyncAgent, RunBudget
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.observability import InMemoryObservabilityLogger
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import AsyncMockLLMClient, AsyncMockTool, tool_response

pytestmark = pytest.mark.asyncio


def _looping_response(messages, tools, system) -> LLMResponse:
    response = tool_response(name="echo", arguments={})
    response.usage = {"input_tokens": 10, "output_tokens": 5}
    return response


def _agent(
    run_budget: RunBudget | None,
    max_iterations: int = 60,
) -> tuple[AsyncAgent, AsyncMockLLMClient, InMemoryObservabilityLogger]:
    llm = AsyncMockLLMClient([_looping_response] * 100)
    logger = InMemoryObservabilityLogger()
    agent = AsyncAgent(
        llm=llm,
        tools=[AsyncMockTool(name="echo", description="Echo.", result="ok")],
        permission_policy=PermissivePolicy(),
        observability_logger=logger,
        run_budget=run_budget,
        max_iterations=max_iterations,
    )
    return agent, llm, logger


async def test_call_limit_stops_loop_gracefully() -> None:
    budget = RunBudget(max_llm_calls=3)
    agent, llm, logger = _agent(budget)
    final = await agent.run("loop forever")

    assert final.startswith("Run stopped: budget exceeded")
    assert "3/3 calls used" in final
    assert len(llm.calls) == 3
    exceeded = logger.get_events("agent.budget_exceeded")
    assert len(exceeded) == 1
    assert exceeded[0]["payload"]["used_calls"] == 3
    assert logger.get_events("agent.run_finished")
    assert not logger.get_events("agent.error")


async def test_token_limit_stops_loop() -> None:
    budget = RunBudget(max_total_tokens=30)
    agent, llm, _ = _agent(budget)
    final = await agent.run("loop forever")

    assert "token budget exhausted" in final
    assert len(llm.calls) == 2
    assert budget.used_tokens == 30


async def test_no_budget_uses_max_iterations_safely() -> None:
    agent, llm, logger = _agent(None, max_iterations=7)
    await agent.run("loop forever")
    assert len(llm.calls) == 7
    assert not logger.get_events("agent.budget_exceeded")
    assert logger.get_events("agent.error")


async def test_budget_accumulates_over_turns_without_reset() -> None:
    budget = RunBudget(max_llm_calls=5)
    agent, _, _ = _agent(budget)
    await agent.run("loop until blocked")
    assert budget.used_calls == 5
    assert budget.used_tokens == 75
