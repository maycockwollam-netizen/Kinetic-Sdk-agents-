"""Tests for the shipped LLMAnswerVerifier + the agent's self-correction loop."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent import Agent, LLMAnswerVerifier
from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.agent.planning import StaticPlanStrategy
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import (
    AsyncMockLLMClient,
    MockLLMClient,
    text_response,
)
from kinetic_sdk.tool import Tool


def _judge(*replies: str) -> MockLLMClient:
    return MockLLMClient([text_response(reply) for reply in replies])


def test_verifier_accepts_rejected_and_unparseable_replies() -> None:
    verifier = LLMAnswerVerifier(_judge("ACCEPTED: complete"), max_tokens=100)
    assert verifier.verify("task", "answer", None).accepted is True

    verifier = LLMAnswerVerifier(_judge("REJECTED: missing test evidence"))
    verdict = verifier.verify("task", "answer", None)
    assert verdict.accepted is False
    assert verdict.feedback == "missing test evidence"

    verifier = LLMAnswerVerifier(_judge("maybe??"))
    verdict = verifier.verify("task", "answer", None)
    assert verdict.accepted is False
    assert "unparseable" in verdict.feedback


def test_verifier_uses_plan_context_and_forwards_max_tokens() -> None:
    judge = MockLLMClient([text_response("ACCEPTED: ok")])
    verifier = LLMAnswerVerifier(judge, max_tokens=77)
    plan = StaticPlanStrategy(["step one", "step two"], goal="goal").create_plan("task", [])

    assert verifier.verify("task", "answer", plan).accepted is True
    sent = judge.calls[0]["messages"][0]["content"]
    assert "step one" in sent and "goal" in sent
    assert judge.calls[0]["kwargs"].get("max_tokens") == 77


def test_verifier_redacts_secrets_before_sending() -> None:
    judge = MockLLMClient([text_response("ACCEPTED: ok")])
    verifier = LLMAnswerVerifier(judge)
    secret = "ghp_" + "a1B2c3D4" * 5

    assert verifier.verify(f"use token {secret}", f"call {secret}", None).accepted is True
    sent = judge.calls[0]["messages"][0]["content"]
    assert secret not in sent
    assert "[REDACTED]" in sent


def test_verifier_propagates_model_errors() -> None:
    from kinetic_sdk.agent.planning import VerificationResult

    class ExplodingLLM:
        def chat(self, messages=None, tools=None, system=None, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("offline")

    verifier = LLMAnswerVerifier(ExplodingLLM())
    with pytest.raises(RuntimeError, match="offline"):
        verifier.verify("task", "answer", None)
    assert isinstance(VerificationResult(accepted=True), VerificationResult)


def test_agent_self_corrects_a_rejected_answer() -> None:
    actor = MockLLMClient([text_response("bad answer"), text_response("good answer")])
    judge = _judge("REJECTED: lacks test evidence", "ACCEPTED: evidence present")
    agent = Agent(
        actor,
        permission_policy=PermissivePolicy(),
        answer_verifier=LLMAnswerVerifier(judge),
        max_verification_retries=1,
    )

    assert agent.run("prove it") == "good answer"
    assert len(actor.calls) == 2
    assert len(judge.calls) == 2
    # The rejection feedback reached the model as a correction round.
    history = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in agent.state.messages
    )
    assert "lacks test evidence" in history


def test_agent_raises_when_corrections_run_out() -> None:
    from kinetic_sdk.agent import AnswerNotVerifiedError

    actor = MockLLMClient([text_response("still bad")])
    judge = _judge("REJECTED: never good")
    agent = Agent(
        actor,
        permission_policy=PermissivePolicy(),
        answer_verifier=LLMAnswerVerifier(judge),
        max_verification_retries=0,
    )
    with pytest.raises(AnswerNotVerifiedError):
        agent.run("prove it")


@pytest.mark.asyncio
async def test_async_agent_self_corrects_a_rejected_answer() -> None:
    actor = AsyncMockLLMClient([text_response("bad answer"), text_response("good answer")])
    # The LLMAnswerVerifier is synchronous; the async agent bridges it via to_thread.
    judge = _judge("REJECTED: lacks test evidence", "ACCEPTED: evidence present")
    agent = AsyncAgent(
        actor,
        permission_policy=PermissivePolicy(),
        answer_verifier=LLMAnswerVerifier(judge),
        max_verification_retries=1,
    )

    assert await agent.run("prove it") == "good answer"
    assert len(actor.calls) == 2
    assert len(judge.calls) == 2


# --- reflection on tool errors ------------------------------------------

def _tool_turn(name: str) -> object:
    from kinetic_sdk.testing import tool_response

    return tool_response("c1", name, {})


def _failing_tool() -> Tool:
    from kinetic_sdk.tool import Tool, ToolResult

    class FailingTool(Tool):
        name = "failing_tool"
        description = "always fails"
        parameters = {"type": "object", "properties": {}, "required": []}

        def execute(self, **params: object) -> ToolResult:
            return ToolResult(error="boom")

    return FailingTool()


def test_agent_adds_reflection_instruction_after_tool_error() -> None:
    from kinetic_sdk.testing import MockTool

    tool = MockTool(name="ok_tool", result="fine")
    llm = MockLLMClient([_tool_turn("failing_tool"), text_response("done")])
    agent = Agent(
        llm,
        tools=[tool, _failing_tool()],
        permission_policy=PermissivePolicy(),
    )
    assert agent.run("go") == "done"
    assert "REFLECTION" in agent.llm.calls[1]["system"]


def test_agent_omits_reflection_after_successful_tool_turn() -> None:
    from kinetic_sdk.testing import MockTool

    llm = MockLLMClient([_tool_turn("ok_tool"), text_response("done")])
    agent = Agent(
        llm,
        tools=[MockTool(name="ok_tool", result="fine")],
        permission_policy=PermissivePolicy(),
    )
    assert agent.run("go") == "done"
    assert "REFLECTION" not in (agent.llm.calls[1]["system"] or "")


def test_agent_reflection_can_be_disabled() -> None:
    llm = MockLLMClient([_tool_turn("failing_tool"), text_response("done")])
    agent = Agent(
        llm,
        tools=[_failing_tool()],  # type: ignore[arg-type]
        permission_policy=PermissivePolicy(),
        reflection_on_error=False,
    )
    assert agent.run("go") == "done"
    assert "REFLECTION" not in (agent.llm.calls[1]["system"] or "")


@pytest.mark.asyncio
async def test_async_agent_adds_reflection_instruction_after_tool_error() -> None:
    llm = AsyncMockLLMClient([_tool_turn("failing_tool"), text_response("done")])
    agent = AsyncAgent(
        llm,
        tools=[_failing_tool()],  # type: ignore[arg-type]  # sync tool bridged
        permission_policy=PermissivePolicy(),
    )
    assert await agent.run("go") == "done"
    assert "REFLECTION" in llm.calls[1]["system"]
