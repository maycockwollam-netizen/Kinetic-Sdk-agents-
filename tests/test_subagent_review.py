"""Tests for subagent/review.py — produce_and_review."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import SpawnBudget, SubagentSpec, produce_and_review
from kinetic_sdk.testing import MockLLMClient, text_response

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["approved"],
    "additionalProperties": True,
}


def _last_user(messages) -> str:
    return next(
        m["content"]
        for m in reversed(messages)
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )


def test_produce_review_revises_until_approved() -> None:
    def handler(messages, tools, system):  # type: ignore[no-untyped-def]
        last = _last_user(messages)
        if last.startswith("CRITIC:"):
            return (
                text_response('{"approved": true, "feedback": "ok"}')
                if "v2" in last
                else text_response('{"approved": false, "feedback": "too short"}')
            )
        return text_response("v2") if "revise" in last.lower() else text_response("v1")

    producer = SubagentSpec(name="producer", system_prompt="p", description="d")
    critic = SubagentSpec(
        name="critic", system_prompt="p", description="d", result_schema=REVIEW_SCHEMA
    )
    parent = Agent(llm=MockLLMClient([handler] * 20), permission_policy=PermissivePolicy())

    out = produce_and_review(
        parent, producer, critic, "write an essay",
        criteria="at least 10 words", budget=SpawnBudget(100), max_revisions=2,
    )
    assert out.result == "v2"
    assert out.approved is True
    assert out.revisions == 1
    assert len(out.reviews) == 2
    assert out.last_review_feedback == "ok"


def test_produce_review_exhausts_revisions_when_never_approved() -> None:
    def handler(messages, tools, system):  # type: ignore[no-untyped-def]
        last = _last_user(messages)
        if last.startswith("CRITIC:"):
            return text_response('{"approved": false, "feedback": "no"}')
        return text_response("draft")

    producer = SubagentSpec(name="producer", system_prompt="p", description="d")
    critic = SubagentSpec(
        name="critic", system_prompt="p", description="d", result_schema=REVIEW_SCHEMA
    )
    parent = Agent(llm=MockLLMClient([handler] * 20), permission_policy=PermissivePolicy())

    out = produce_and_review(
        parent, producer, critic, "t", budget=SpawnBudget(100), max_revisions=2
    )
    assert out.approved is False
    assert out.revisions == 2
    assert len(out.reviews) == 3
    assert out.result == "draft"