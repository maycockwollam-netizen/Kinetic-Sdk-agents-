"""Tests for subagent/budget.py — SpawnBudget + RepetitionCircuitBreaker."""

from __future__ import annotations

import threading

import pytest

from kinetic_sdk.subagent import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    DEFAULT_MAX_TOTAL_TOOL_CALLS,
    BudgetExceededError,
    RepetitionCircuitBreaker,
    RepetitionLimitError,
    SpawnBudget,
)


class TestSpawnBudget:
    def test_defaults(self) -> None:
        budget = SpawnBudget()
        assert budget.max_total_tool_calls == DEFAULT_MAX_TOTAL_TOOL_CALLS
        assert budget.used == 0
        assert budget.remaining == DEFAULT_MAX_TOTAL_TOOL_CALLS
        assert not budget.exhausted

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "4000"])
    def test_invalid_cap_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_total_tool_calls"):
            SpawnBudget(bad)  # type: ignore[arg-type]

    def test_exactly_cap_calls_allowed_then_raises(self) -> None:
        budget = SpawnBudget(max_total_tool_calls=3)
        for _ in range(3):
            budget.record_tool_call("agent-a", "echo")
        assert budget.used == 3
        assert budget.remaining == 0
        assert budget.exhausted
        with pytest.raises(BudgetExceededError, match="3/3"):
            budget.record_tool_call("agent-a", "echo")
        # The failed call is NOT charged: the counter never exceeds the cap.
        assert budget.used == 3

    def test_one_instance_shared_across_agents_accumulates(self) -> None:
        # Simulates a 3-level tree: root sub-agent, child, grandchild all
        # charging the SAME budget instance.
        budget = SpawnBudget(max_total_tool_calls=10)
        budget.record_tool_call("level-1", "delegate")
        budget.record_tool_call("level-2", "echo")
        budget.record_tool_call("level-2", "delegate")
        budget.record_tool_call("level-3", "echo")
        assert budget.used == 4
        assert budget.calls_by_agent() == {"level-1": 1, "level-2": 2, "level-3": 1}

    def test_thread_safety_no_lost_updates(self) -> None:
        budget = SpawnBudget(max_total_tool_calls=10_000)

        def worker(agent: str) -> None:
            for _ in range(1000):
                budget.record_tool_call(agent, "echo")

        threads = [threading.Thread(target=worker, args=(f"a{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert budget.used == 10_000
        assert sum(budget.calls_by_agent().values()) == 10_000

    def test_thread_safety_raises_exactly_overflow_times(self) -> None:
        budget = SpawnBudget(max_total_tool_calls=50)

        def worker(agent: str, raised: list[int], idx: int) -> None:
            count = 0
            for _ in range(10):
                try:
                    budget.record_tool_call(agent, "echo")
                except BudgetExceededError:
                    count += 1
            raised[idx] = count

        raised = [0] * 10
        threads = [
            threading.Thread(target=worker, args=(f"a{i}", raised, i)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 100 attempts total: exactly 50 charged, exactly 50 rejected.
        assert budget.used == 50
        assert sum(raised) == 50


class TestRepetitionCircuitBreaker:
    def test_defaults(self) -> None:
        breaker = RepetitionCircuitBreaker()
        assert breaker.max_consecutive_repeats == DEFAULT_MAX_CONSECUTIVE_REPEATS
        assert breaker.consecutive_count == 0

    @pytest.mark.parametrize("bad", [0, -3, 2.5, "20"])
    def test_invalid_limit_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_consecutive_repeats"):
            RepetitionCircuitBreaker(bad)  # type: ignore[arg-type]

    def test_identical_call_trip_at_limit_plus_one(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=3)
        for _ in range(3):
            breaker.record("echo", {"message": "hi"})
        assert breaker.consecutive_count == 3
        with pytest.raises(RepetitionLimitError, match="identical arguments 4 times"):
            breaker.record("echo", {"message": "hi"})

    def test_same_tool_different_arguments_resets(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        breaker.record("echo", {"message": "a"})
        breaker.record("echo", {"message": "a"})
        breaker.record("echo", {"message": "b"})  # different args -> reset
        breaker.record("echo", {"message": "b"})
        breaker.record("echo", {"message": "a"})  # back to a -> reset again
        assert breaker.consecutive_count == 1

    def test_different_tool_interleaved_resets(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        breaker.record("echo", {"message": "a"})
        breaker.record("echo", {"message": "a"})
        breaker.record("status", {})  # different tool -> reset
        breaker.record("echo", {"message": "a"})
        breaker.record("echo", {"message": "a"})
        assert breaker.consecutive_count == 2

    def test_argument_key_ordering_is_irrelevant(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        breaker.record("tool", {"a": 1, "b": 2})
        breaker.record("tool", {"b": 2, "a": 1})
        with pytest.raises(RepetitionLimitError):
            breaker.record("tool", {"a": 1, "b": 2})

    def test_none_and_non_serialisable_arguments_handled(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        opaque = object()  # not JSON-serialisable -> falls back to str()
        breaker.record("tool", None)
        breaker.record("tool", None)
        with pytest.raises(RepetitionLimitError):
            breaker.record("tool", None)
        breaker.record("tool", {"obj": opaque})
        breaker.record("tool", {"obj": opaque})
        with pytest.raises(RepetitionLimitError):
            breaker.record("tool", {"obj": opaque})

    def test_two_breakers_track_agents_independently(self) -> None:
        breaker_a = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        breaker_b = RepetitionCircuitBreaker(max_consecutive_repeats=2)
        breaker_a.record("echo", {"message": "x"})
        breaker_a.record("echo", {"message": "x"})
        # Agent B repeats the SAME call: its own counter, unaffected by A.
        breaker_b.record("echo", {"message": "x"})
        assert breaker_b.consecutive_count == 1
        with pytest.raises(RepetitionLimitError):
            breaker_a.record("echo", {"message": "x"})
        breaker_b.record("echo", {"message": "x"})  # B still fine at 2

    def test_reset_clears_state(self) -> None:
        breaker = RepetitionCircuitBreaker(max_consecutive_repeats=1)
        breaker.record("echo", {})
        breaker.reset()
        assert breaker.consecutive_count == 0
        breaker.record("echo", {})  # would have tripped without the reset
