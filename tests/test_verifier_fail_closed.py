"""Verifier fail-closed behaviour (sync + async parity).

Covers the audit finding that a verifier which raises used to silently
*accept* the final answer. With ``verifier_fail_closed=True`` (default) an
exception is treated as a rejection, so the agent raises
``AnswerNotVerifiedError`` instead of approving unverified output.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent import Agent, AnswerNotVerifiedError
from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import AsyncMockLLMClient, MockLLMClient, text_response


class RaisingVerifier:
    def verify(self, task, answer, plan):
        raise RuntimeError("verifier exploded")


def _agent(**kwargs):
    return Agent(
        llm=MockLLMClient([text_response("answer")]),
        permission_policy=PermissivePolicy(),
        **kwargs,
    )


def test_verifier_fail_closed_rejects_on_exception():
    agent = _agent(answer_verifier=RaisingVerifier(), max_verification_retries=0)
    with pytest.raises(AnswerNotVerifiedError):
        agent.run("task")


def test_verifier_fail_open_accepts_on_exception():
    agent = _agent(
        answer_verifier=RaisingVerifier(),
        max_verification_retries=0,
        verifier_fail_closed=False,
    )
    assert agent.run("task") == "answer"


@pytest.mark.asyncio
async def test_async_verifier_fail_closed_rejects_on_exception():
    agent = AsyncAgent(
        llm=AsyncMockLLMClient([text_response("answer")]),
        permission_policy=PermissivePolicy(),
        answer_verifier=RaisingVerifier(),
        max_verification_retries=0,
    )
    with pytest.raises(AnswerNotVerifiedError):
        await agent.run("task")
