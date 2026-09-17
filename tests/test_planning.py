"""Plan/execute/verify integration coverage for the synchronous agent."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent import (
    AgentMode,
    AnswerNotVerifiedError,
    Classification,
    Plan,
    StaticPlanStrategy,
    TaskComplexity,
    VerificationResult,
)
from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.security.policy import PermissivePolicy
from tests._helpers import EchoTool, MockLLM, text_response


class RejectOnceVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Plan | None]] = []

    def verify(self, task: str, answer: str, plan: Plan | None) -> VerificationResult:
        self.calls.append((task, answer, plan))
        if len(self.calls) == 1:
            return VerificationResult(False, "Include the observed evidence.")
        return VerificationResult(True)


class RejectingVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Plan | None]] = []

    def verify(self, task: str, answer: str, plan: Plan | None) -> VerificationResult:
        self.calls.append((task, answer, plan))
        return VerificationResult(False, "The answer lacks supporting evidence.")


class FlashClassifier:
    alias = "kinetic-classifier-v1"

    def classify(self, task: str) -> Classification:
        del task
        return Classification(
            complexity=TaskComplexity.SIMPLE,
            mode=AgentMode.FLASH,
            confidence=1.0,
            rationale="test",
        )


def test_plan_is_injected_and_failed_verification_gets_one_correction_turn():
    llm = MockLLM([text_response("unverified"), text_response("verified evidence")])
    events: list[str] = []
    bus = EventBus()
    bus.subscribe("*", lambda event: events.append(event.type))
    verifier = RejectOnceVerifier()
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        permission_policy=PermissivePolicy(),
        event_bus=bus,
        planner=StaticPlanStrategy(["Inspect evidence", "Report verified result"]),
        answer_verifier=verifier,
    )

    assert agent.run("Solve the task") == "verified evidence"
    assert agent.plan is not None
    assert agent.plan.steps == ("Inspect evidence", "Report verified result")
    assert "Execution plan" in llm.calls[0]["system"]
    assert "Verifier feedback" in llm.calls[1]["messages"][-1]["content"]
    assert [call[1] for call in verifier.calls] == ["unverified", "verified evidence"]
    assert "agent.plan_created" in events
    assert "agent.verification_retry" in events
    assert "agent.verification_passed" in events


def test_planner_failure_fails_open_without_changing_normal_run():
    class BrokenPlanner:
        def create_plan(self, task: str, tools: list[dict[str, object]]) -> Plan:
            raise RuntimeError("unavailable")

    llm = MockLLM([text_response("normal answer")])
    agent = Agent(llm=llm, planner=BrokenPlanner())

    assert agent.run("task") == "normal answer"
    assert agent.plan is None


def test_repeated_verification_rejection_escalates_flash_to_max_before_budget_runs_out():
    verifier = RejectingVerifier()
    llm = MockLLM([text_response(f"rejected answer {index}") for index in range(11)])
    agent = Agent(
        llm=llm,
        classifier=FlashClassifier(),
        answer_verifier=verifier,
        max_verification_retries=10,
    )

    with pytest.raises(AnswerNotVerifiedError):
        agent.run("Solve the task")

    assert agent.mode is AgentMode.MAX
    assert agent.max_iterations == Agent.MODE_MAX_ITERATIONS[AgentMode.MAX]
    assert len(verifier.calls) == 11


def test_verification_retry_exhaustion_raises_instead_of_returning_rejected_answer():
    verifier = RejectingVerifier()
    agent = Agent(
        llm=MockLLM([text_response("rejected answer")]),
        answer_verifier=verifier,
        max_verification_retries=0,
    )

    with pytest.raises(AnswerNotVerifiedError) as exc_info:
        agent.run("Solve the task")

    assert exc_info.value.answer == "rejected answer"
    assert exc_info.value.feedback == "The answer lacks supporting evidence."
    assert len(verifier.calls) == 1


def test_iteration_cap_raises_when_a_verification_retry_is_pending():
    verifier = RejectingVerifier()
    agent = Agent(
        llm=MockLLM([text_response("rejected answer")]),
        classifier=FlashClassifier(),
        answer_verifier=verifier,
        max_verification_retries=1,
        max_iterations=1,
    )

    with pytest.raises(AnswerNotVerifiedError):
        agent.run("Solve the task")

    assert agent.mode is AgentMode.MAX
