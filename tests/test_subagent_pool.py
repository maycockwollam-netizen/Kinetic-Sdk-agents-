"""Tests for bounded parallel sub-agent orchestration."""

from __future__ import annotations

import threading
import time

import pytest

from kinetic_sdk.agent.agent import Agent, AnswerNotVerifiedError
from kinetic_sdk.agent.planning import VerificationResult
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    SpawnBudget,
    SubagentPool,
    SubagentSpec,
    spawn_subagent_pool,
)
from kinetic_sdk.subagent.delegation import DelegationResult
from kinetic_sdk.testing import MockLLMClient


def _spec(name: str) -> SubagentSpec:
    return SubagentSpec(name=name, system_prompt="worker", description=f"task {name}")


def test_pool_limits_workers_and_preserves_spec_order(monkeypatch: pytest.MonkeyPatch) -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(parent, spec, task, budget):  # type: ignore[no-untyped-def]
        nonlocal running, peak
        assert task == spec.description
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03)
        with lock:
            running -= 1
        return DelegationResult(spec.name, spec.name + "-id")

    monkeypatch.setattr("kinetic_sdk.subagent.pool.run_subagent", fake_run)
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    results = SubagentPool(2).run_all(parent, [_spec("one"), _spec("two"), _spec("three")], SpawnBudget())
    assert [result.final_message for result in results] == ["one", "two", "three"]
    assert peak == 2


def test_pool_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="positive"):
        SubagentPool(0)


def test_spawn_pool_default_merge_and_existing_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_all(self, parent, specs, budget):  # type: ignore[no-untyped-def]
        return [DelegationResult("first", "a"), DelegationResult("second", "b")]

    monkeypatch.setattr(SubagentPool, "run_all", fake_run_all)
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    assert spawn_subagent_pool(parent, [_spec("one"), _spec("two")]) == (
        "first\n\n--- Sub-agent result ---\n\nsecond"
    )


def test_spawn_pool_raises_existing_verification_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SubagentPool,
        "run_all",
        lambda self, parent, specs, budget: [DelegationResult("wrong", "a")],
    )
    class RejectingVerifier:
        def verify(self, task, answer, plan):  # type: ignore[no-untyped-def]
            return VerificationResult(False, "missing proof")

    parent = Agent(
        llm=MockLLMClient([]),
        permission_policy=PermissivePolicy(),
        answer_verifier=RejectingVerifier(),
    )
    with pytest.raises(AnswerNotVerifiedError, match="rejected"):
        spawn_subagent_pool(parent, [_spec("one")])
