"""Unit + integration tests for the root run budget (``agent/budget.py``).

``RunBudget`` mirrors the ``SpawnBudget`` pattern from ``subagent/``: the
check raises internally, and the agent loop converts it into a graceful
stop (``agent.budget_exceeded`` event + explanatory final text) so a budget
trip never crashes a run.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent import Agent, RunBudget, RunBudgetExceeded
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.observability import InMemoryObservabilityLogger
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, tool_response


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


class TestRunBudgetUnit:
    def test_no_limits_never_blocks(self) -> None:
        budget = RunBudget()
        for _ in range(100):
            budget.record(_usage(500, 500))
            budget.check()  # must never raise
        assert budget.used_calls == 100
        assert budget.used_tokens == 100_000
        assert budget.exhausted is False

    def test_call_limit_trips_exactly_at_cap(self) -> None:
        budget = RunBudget(max_llm_calls=3)
        for _ in range(3):
            budget.record(None)
        assert budget.used_calls == 3
        assert budget.exhausted is True
        with pytest.raises(RunBudgetExceeded, match="call budget exhausted"):
            budget.check()

    def test_token_limit_trips(self) -> None:
        budget = RunBudget(max_total_tokens=100)
        budget.record(_usage(60, 40))
        with pytest.raises(RunBudgetExceeded, match="token budget exhausted"):
            budget.check()
        assert budget.exhausted is True

    def test_both_limits_call_limit_reported_first(self) -> None:
        budget = RunBudget(max_llm_calls=2, max_total_tokens=2)
        budget.record(_usage(1, 1))
        budget.record(_usage(1, 1))
        # Both limits are simultaneously exhausted; the CALL reason wins.
        with pytest.raises(RunBudgetExceeded, match="call budget exhausted"):
            budget.check()

    def test_tokens_accumulate_across_records(self) -> None:
        budget = RunBudget(max_total_tokens=1_000)
        budget.record(_usage(100, 50))
        budget.record(_usage(70, 30))
        budget.record(_usage(40, 100))
        assert budget.used_tokens == 390
        assert budget.used_calls == 3

    def test_usage_without_token_numbers_still_counts_the_call(self) -> None:
        budget = RunBudget(max_llm_calls=2)
        budget.record({})
        budget.record(None)
        assert budget.used_calls == 2
        assert budget.used_tokens == 0
        with pytest.raises(RunBudgetExceeded):
            budget.check()

    def test_rejects_invalid_constructor_values(self) -> None:
        with pytest.raises(ValueError):
            RunBudget(max_llm_calls=0)
        with pytest.raises(ValueError):
            RunBudget(max_total_tokens=-5)
        with pytest.raises(ValueError):
            RunBudget(max_llm_calls="3")  # type: ignore[arg-type]

    def test_details_snapshot(self) -> None:
        budget = RunBudget(max_llm_calls=10, max_total_tokens=100)
        budget.record(_usage(10, 5))
        assert budget.details() == {
            "max_llm_calls": 10,
            "max_total_tokens": 100,
            "used_calls": 1,
            "used_tokens": 15,
        }


def _looping_response(messages, tools, system) -> LLMResponse:
    """Every turn requests the same tool call and reports 10+5 tokens."""
    response = tool_response(name="echo", arguments={})
    response.usage = _usage(10, 5)
    return response


class TestAgentIntegration:
    def _agent(
        self,
        run_budget: RunBudget | None,
        max_iterations: int = 60,
        script_length: int = 100,
    ) -> tuple[Agent, MockLLMClient, InMemoryObservabilityLogger]:
        llm = MockLLMClient([_looping_response] * script_length)
        logger = InMemoryObservabilityLogger()
        agent = Agent(
            llm=llm,
            tools=[MockTool(name="echo", description="Echo.", result="ok")],
            permission_policy=PermissivePolicy(),
            observability_logger=logger,
            run_budget=run_budget,
            max_iterations=max_iterations,
        )
        return agent, llm, logger

    def test_call_limit_stops_loop_gracefully(self) -> None:
        budget = RunBudget(max_llm_calls=3)
        agent, llm, logger = self._agent(budget)
        final = agent.run("loop forever")

        assert final.startswith("Run stopped: budget exceeded")
        assert "3/3 calls used" in final
        # The 3rd call still ran; the 4th was blocked before it started.
        assert len(llm.calls) == 3
        assert budget.used_calls == 3
        exceeded = logger.get_events("agent.budget_exceeded")
        assert len(exceeded) == 1
        payload = exceeded[0]["payload"]
        assert payload["max_llm_calls"] == 3
        assert payload["used_calls"] == 3
        assert payload["used_tokens"] == 45  # 3 calls x (10 in + 5 out)
        # The run still finished cleanly, not crashed.
        assert logger.get_events("agent.run_finished")
        assert not logger.get_events("agent.error")

    def test_token_limit_stops_loop(self) -> None:
        budget = RunBudget(max_total_tokens=30)
        agent, llm, logger = self._agent(budget)
        final = agent.run("loop forever")

        assert "token budget exhausted" in final
        # 2 calls x 15 tokens = 30 hits the cap; the 3rd call is blocked.
        assert len(llm.calls) == 2
        assert budget.used_tokens == 30
        assert logger.get_events("agent.budget_exceeded")

    def test_both_limits_configured(self) -> None:
        budget = RunBudget(max_llm_calls=5, max_total_tokens=25)
        agent, llm, _ = self._agent(budget)
        final = agent.run("loop forever")
        # Token limit (25) trips before the call limit (5): call 1 uses 15
        # tokens (passes), call 2 brings the total to 30 (blocked next check).
        assert "token budget exhausted" in final
        assert len(llm.calls) == 2

    def test_no_budget_uses_max_iterations_safely(self) -> None:
        agent, llm, logger = self._agent(None, max_iterations=7)
        agent.run("loop forever")
        assert len(llm.calls) == 7
        assert not logger.get_events("agent.budget_exceeded")
        assert logger.get_events("agent.error")  # max_iterations, as before

    def test_budget_accumulates_over_turns_without_reset(self) -> None:
        budget = RunBudget(max_llm_calls=5)
        agent, _, _ = self._agent(budget)
        agent.run("loop until blocked")
        # After the run, the budget object reflects the FULL accumulation —
        # it must not reset between LLM turns of the run.
        assert budget.used_calls == 5
        assert budget.used_tokens == 75
        assert agent.run_budget is budget

    def test_structured_usage_accumulator_still_tracks_total(self) -> None:
        budget = RunBudget(max_total_tokens=30)
        agent, _, _ = self._agent(budget)
        agent.run("loop")
        snapshot = agent.usage.snapshot()
        assert snapshot.calls == 2
        assert snapshot.input_tokens == 20
        assert snapshot.output_tokens == 10
