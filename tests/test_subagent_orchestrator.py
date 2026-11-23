"""Tests for the SubagentOrchestrator (router + streaming + retry)."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    OrchestrationResult,
    OrchestrationTask,
    SpawnBudget,
    SubagentOrchestrator,
    SubagentSpec,
)
from kinetic_sdk.testing import MockLLMClient, text_response

SPECS = {
    "alpha": SubagentSpec(name="alpha", system_prompt="p", description="d", model="alpha-model"),
    "beta": SubagentSpec(name="beta", system_prompt="p", description="d", model="beta-model"),
}


def _parent() -> Agent:
    return Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())


def _factory(model: str) -> MockLLMClient:
    if model == "alpha-model":
        return MockLLMClient([text_response("alpha done")])
    return MockLLMClient([text_response("beta done")])


def test_routes_by_name_and_preserves_input_order() -> None:
    orch = SubagentOrchestrator(_parent(), SPECS, max_concurrency=2)
    results = orch.run(
        [
            OrchestrationTask(task="t1", subagent="alpha"),
            OrchestrationTask(task="t2", subagent="beta"),
            OrchestrationTask(task="t3", subagent="alpha"),
        ],
        budget=SpawnBudget(100),
        llm_factory=_factory,
    )
    assert [r.subagent for r in results] == ["alpha", "beta", "alpha"]
    assert all(r.success for r in results)
    assert [r.final_message for r in results] == ["alpha done", "beta done", "alpha done"]
    assert all(r.attempts == 1 for r in results)


def test_failure_of_one_task_does_not_block_others() -> None:
    specs = {
        "bad": SubagentSpec(name="bad", system_prompt="p", description="d", model="bad"),
        "good": SubagentSpec(name="good", system_prompt="p", description="d", model="good"),
    }

    def factory(model: str) -> MockLLMClient:
        if model == "bad":
            def boom(messages, tools, system):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

            return MockLLMClient([boom])
        return MockLLMClient([text_response("good done")])

    results = SubagentOrchestrator(_parent(), specs, max_concurrency=2).run(
        [
            OrchestrationTask(task="a", subagent="bad"),
            OrchestrationTask(task="b", subagent="good"),
        ],
        budget=SpawnBudget(100),
        llm_factory=factory,
    )
    assert results[0].success is False
    assert "boom" in results[0].error
    assert results[1].success is True
    assert results[1].final_message == "good done"


def test_streaming_events_cover_start_and_finish() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("orchestrator.task_started", lambda event: seen.append(event.type))
    bus.subscribe("orchestrator.task_finished", lambda event: seen.append(event.type))
    orch = SubagentOrchestrator(_parent(), SPECS, max_concurrency=2, event_bus=bus)
    orch.run(
        [
            OrchestrationTask(task="t1", subagent="alpha"),
            OrchestrationTask(task="t2", subagent="beta"),
        ],
        budget=SpawnBudget(100),
        llm_factory=_factory,
    )
    assert seen.count("orchestrator.task_started") == 2
    assert seen.count("orchestrator.task_finished") == 2


def test_retry_on_failure_runs_worker_twice() -> None:
    state = {"spawns": 0}
    specs = {"flaky": SubagentSpec(name="flaky", system_prompt="p", description="d", model="flaky")}

    def factory(model: str) -> MockLLMClient:
        if state["spawns"] == 0:
            state["spawns"] += 1

            def boom(messages, tools, system):  # type: ignore[no-untyped-def]
                raise RuntimeError("transient")

            return MockLLMClient([boom])
        state["spawns"] += 1
        return MockLLMClient([text_response("retried ok")])

    bus = EventBus()
    retries: list[str] = []
    bus.subscribe("orchestrator.task_retry", lambda event: retries.append(event.type))
    orch = SubagentOrchestrator(_parent(), specs, max_concurrency=1, event_bus=bus)
    results = orch.run(
        [OrchestrationTask(task="t", subagent="flaky")],
        budget=SpawnBudget(100),
        llm_factory=factory,
        retry_on_failure=1,
    )
    assert results[0].success is True
    assert results[0].final_message == "retried ok"
    assert results[0].attempts == 2
    assert retries == ["orchestrator.task_retry"]


def test_unknown_subagent_name_rejected() -> None:
    orch = SubagentOrchestrator(_parent(), SPECS, max_concurrency=1)
    with pytest.raises(ValueError, match="unknown sub-agent"):
        orch.run(
            [OrchestrationTask(task="x", subagent="nope")],
            budget=SpawnBudget(100),
            llm_factory=_factory,
        )


def test_result_to_dict_shape() -> None:
    result = OrchestrationResult(
        OrchestrationTask(task="t", subagent="alpha"), success=True, final_message="m", agent_id="a-1"
    )
    assert result.to_dict()["subagent"] == "alpha"
    assert result.to_dict()["final_message"] == "m"


def test_orchestrator_timeout_returns_failed_results_for_pending() -> None:
    import time

    from kinetic_sdk.testing import text_response as _tr

    specs = {"slow": SubagentSpec(name="slow", system_prompt="p", description="d", model="slow")}

    def slow_llm(model: str) -> MockLLMClient:
        def slow(messages, tools, system):  # type: ignore[no-untyped-def]
            time.sleep(0.05)
            return _tr("done")

        return MockLLMClient([slow])

    results = SubagentOrchestrator(_parent(), specs, max_concurrency=1).run(
        [OrchestrationTask(task="t", subagent="slow")],
        budget=SpawnBudget(100),
        llm_factory=slow_llm,
        timeout=0.005,
    )
    # Timeout fired -> the pending task still yields a failed result (no missing key).
    assert len(results) == 1
    assert results[0].success is False
    assert "timed out" in results[0].error
