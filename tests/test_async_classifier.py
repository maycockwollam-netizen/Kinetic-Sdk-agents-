"""Tests for the async task classifiers."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.async_classifier import (
    AsyncDefaultClassifier,
    AsyncLiteLLMClassifier,
)
from kinetic_sdk.agent.classifier import TaskComplexity
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.testing import AsyncMockLLMClient, text_response

pytestmark = pytest.mark.asyncio


async def test_default_classifier_always_max():
    classifier = AsyncDefaultClassifier()
    result = await classifier.classify("anything")
    assert result.mode is AgentMode.MAX
    assert result.complexity is TaskComplexity.COMPLEX


async def test_litellm_classifier_simple_routes_flash():
    client = AsyncMockLLMClient([text_response("SIMPLE")])
    classifier = AsyncLiteLLMClassifier(client=client)
    result = await classifier.classify("what is 2+2?")
    assert result.complexity is TaskComplexity.SIMPLE
    assert result.mode is AgentMode.FLASH
    # The real model name must never leak — only the alias surfaces.
    assert classifier.model == "kinetic-classifier-v1"
    assert classifier.alias == "kinetic-classifier-v1"


async def test_litellm_classifier_complex_routes_max():
    client = AsyncMockLLMClient([text_response("COMPLEX")])
    classifier = AsyncLiteLLMClassifier(client=client)
    result = await classifier.classify("refactor the whole module")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX


async def test_litellm_classifier_unparseable_falls_back_to_complex():
    client = AsyncMockLLMClient([text_response("I have no idea")])
    classifier = AsyncLiteLLMClassifier(client=client)
    result = await classifier.classify("something")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX


async def test_litellm_classifier_exception_falls_back_to_complex():
    async def boom(messages, tools, system):
        raise RuntimeError("provider down")

    client = AsyncMockLLMClient([boom])
    classifier = AsyncLiteLLMClassifier(client=client)
    result = await classifier.classify("something")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX
    assert result.confidence == 0.0
    assert result.rationale == "classifier_error"


async def test_litellm_classifier_prompt_shape():
    client = AsyncMockLLMClient([text_response("COMPLEX")])
    classifier = AsyncLiteLLMClassifier(client=client, summary="earlier context")
    await classifier.classify("the task")
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "SIMPLE" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "the task" in messages[1]["content"]
    assert "earlier context" in messages[1]["content"]
    # One-word answer => tiny token budget.
    assert client.calls[0]["kwargs"]["max_tokens"] == 10


async def test_async_callable_script_entries_are_awaited():
    async def scripted(messages, tools, system):
        return LLMResponse(content="SIMPLE", stop_reason="end_turn")

    client = AsyncMockLLMClient([scripted])
    classifier = AsyncLiteLLMClassifier(client=client)
    result = await classifier.classify("task")
    assert result.mode is AgentMode.FLASH
