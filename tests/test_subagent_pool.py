"""Tests for bounded parallel sub-agent orchestration."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
from kinetic_sdk.subagent.pool import _SerializedLLMClient
from kinetic_sdk.testing import MockLLMClient, text_response


def _spec(name: str) -> SubagentSpec:
    return SubagentSpec(name=name, system_prompt="worker", description=f"task {name}")


def test_pool_limits_workers_and_preserves_spec_order(monkeypatch: pytest.MonkeyPatch) -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(parent, spec, task, budget, **kwargs):  # type: ignore[no-untyped-def]
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
    def fake_run_all(self, parent, specs, budget, **kwargs):  # type: ignore[no-untyped-def]
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
        lambda self, parent, specs, budget, **kwargs: [DelegationResult("wrong", "a")],
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


class _RaceDetectingMockLLMClient(MockLLMClient):
    """A MockLLMClient that makes overlapping calls observable in tests."""

    def __init__(self, responses):  # type: ignore[no-untyped-def]
        super().__init__(responses)
        self._active = 0
        self._active_lock = threading.Lock()
        self.overlapped = False

    def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[no-untyped-def]
        with self._active_lock:
            self._active += 1
            self.overlapped = self.overlapped or self._active > 1
        try:
            # Give every pool worker a chance to enter an unprotected client.
            time.sleep(0.002)
            return super().chat(messages, tools=tools, system=system, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1


def _reply_for_task(messages, tools, system):  # type: ignore[no-untyped-def]
    del tools, system
    return text_response(f"reply-for-{messages[-1]['content']}")


def test_pool_serializes_shared_mock_llm_calls_across_repeated_runs() -> None:
    specs = [_spec(f"worker-{index}") for index in range(8)]
    expected = [f"reply-for-task worker-{index}" for index in range(8)]

    # A separate run repeatedly exercises actual thread contention. Each
    # response callable is input-sensitive, so a worker receiving another
    # worker's turn cannot accidentally satisfy the assertion.
    for _ in range(10):
        shared_llm = _RaceDetectingMockLLMClient([_reply_for_task] * len(specs))
        parent = Agent(llm=shared_llm, permission_policy=PermissivePolicy())
        results = SubagentPool(max_concurrency=len(specs)).run_all(
            parent, specs, SpawnBudget(max_total_tool_calls=1000)
        )
        assert [result.final_message for result in results] == expected
        assert not shared_llm.overlapped


def test_serialized_llm_client_serializes_chat_without_changing_response() -> None:
    inner = _RaceDetectingMockLLMClient([text_response("one"), text_response("two")])
    client = _SerializedLLMClient(inner)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.chat, [{"role": "user", "content": str(index)}])
            for index in range(2)
        ]
        responses = [future.result() for future in futures]

    assert sorted(response.content for response in responses) == ["one", "two"]
    assert not inner.overlapped


def test_pool_forwards_factory_and_builds_client_per_inherited_model() -> None:
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    specs = [_spec("one"), _spec("two")]
    built_models: list[str] = []

    def factory(model: str) -> MockLLMClient:
        built_models.append(model)
        return MockLLMClient([text_response(f"done-{len(built_models)}")], model=model)

    results = SubagentPool(max_concurrency=2).run_all(
        parent, specs, SpawnBudget(), llm_factory=factory
    )

    assert built_models == [parent.llm.model, parent.llm.model]
    assert sorted(result.final_message for result in results) == ["done-1", "done-2"]


def test_spawn_pool_forwards_llm_factory() -> None:
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    built_models: list[str] = []

    def factory(model: str) -> MockLLMClient:
        built_models.append(model)
        return MockLLMClient([text_response("done")], model=model)

    assert spawn_subagent_pool(parent, [_spec("one")], llm_factory=factory) == "done"
    assert built_models == [parent.llm.model]
