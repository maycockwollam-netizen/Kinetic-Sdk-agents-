"""Integration tests against a REAL LLM provider — no mocking of litellm.

These tests verify the parts unit tests can only fake: that
:class:`LiteLLMClient`'s Anthropic<->OpenAI translation layer actually works
against a live provider, that a real model really decides to call a tool when
asked, and that a full :class:`Agent` loop completes end to end.

They are deliberately EXCLUDED from the default ``pytest -q`` run (see
``addopts`` in ``pyproject.toml``) because they need a real API key and a
reachable provider. Run them explicitly::

    KINETIC_INTEGRATION_API_KEY=... pytest -m integration -q

Configuration (all optional except the key):

* ``KINETIC_INTEGRATION_API_KEY`` — API key. Falls back to
  ``OPENHANDS_API_KEY`` for convenience in OpenHands environments.
* ``KINETIC_INTEGRATION_MODEL`` — LiteLLM model string
  (default ``openai/openhands/glm-5.2``).
* ``KINETIC_INTEGRATION_API_BASE`` — custom OpenAI-compatible endpoint
  (default the OpenHands LLM proxy).

The whole module skips when no key is present, so a keyless environment
collects these tests as skips rather than failures. External flakiness (rate
limits, provider outages) can still fail them — that is expected and is why
the CI job for these tests is non-blocking.
"""

from __future__ import annotations

import os

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.llm.client import LiteLLMClient
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockTool

API_KEY = os.environ.get("KINETIC_INTEGRATION_API_KEY") or os.environ.get(
    "OPENHANDS_API_KEY"
)
MODEL = os.environ.get("KINETIC_INTEGRATION_MODEL", "openai/openhands/glm-5.2")
API_BASE = os.environ.get(
    "KINETIC_INTEGRATION_API_BASE", "https://llm-proxy.app.all-hands.dev"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY,
        reason="no KINETIC_INTEGRATION_API_KEY/OPENHANDS_API_KEY set",
    ),
]


@pytest.fixture()
def client() -> LiteLLMClient:
    return LiteLLMClient(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        max_tokens=256,
        timeout=60,
        max_retries=1,
    )


WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def test_simple_chat_round_trip(client: LiteLLMClient) -> None:
    """One complete chat turn: send a message, get a real response back."""
    response = client.chat(
        messages=[{"role": "user", "content": "Reply with exactly the word: PONG"}],
    )
    assert response.content.strip(), "expected non-empty assistant text"
    assert not response.tool_calls
    # The stop-reason mapping must produce the Anthropic-style value.
    assert response.stop_reason == "end_turn"


def test_real_tool_call_and_translation_layer(client: LiteLLMClient) -> None:
    """A real model must request the declared tool, and the tool_result in
    Anthropic block format must round-trip through the OpenAI translation
    layer back to the provider without a 400.
    """
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "What is the weather in Paris right now? "
                "Use the get_weather tool to find out."
            ),
        }
    ]
    first = client.chat(messages=messages, tools=[WEATHER_SCHEMA])
    assert first.tool_calls, "model did not request the tool it was told to use"
    call = first.tool_calls[0]
    assert call.name == "get_weather"
    assert isinstance(call.arguments, dict)
    assert "paris" in str(call.arguments.get("city", "")).lower()
    assert first.stop_reason == "tool_use"

    # Feed the tool result back in Anthropic tool_result block format — this
    # is the exact shape ConversationState stores — and require the provider
    # to accept it (i.e. the OpenAI translation produced a valid history).
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            ],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": "Sunny, 22C",
                }
            ],
        }
    )
    second = client.chat(messages=messages, tools=[WEATHER_SCHEMA])
    assert not second.tool_calls
    assert "22" in second.content or "sunny" in second.content.lower()


def test_streaming_round_trip(client: LiteLLMClient) -> None:
    """chat_stream must yield text deltas then a done event with the full
    aggregated response from the real provider stream."""
    events = list(
        client.chat_stream(
            messages=[
                {"role": "user", "content": "Count from 1 to 5, separated by spaces."}
            ],
        )
    )
    assert events[-1].type == "done"
    text_deltas = [e.delta for e in events if e.type == "text"]
    assert text_deltas, "expected at least one streamed text delta"
    final = events[-1].delta
    assert final is not None
    assert "".join(text_deltas) == final.content
    assert "5" in final.content


def test_agent_loop_with_real_llm(client: LiteLLMClient) -> None:
    """Full Agent.run against the real model: it must call the tool for real
    (executed by the loop, not simulated) and finish with a text answer."""
    weather = MockTool(
        "get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        result="Sunny, 22C",
    )
    agent = Agent(
        llm=client,
        tools=[weather],
        permission_policy=PermissivePolicy(),
        max_iterations=5,
    )
    answer = agent.run("Use the get_weather tool to check the weather in Paris.")
    assert weather.calls, "agent loop never executed the tool"
    assert "paris" in weather.calls[0].get("city", "").lower()
    assert answer.strip(), "expected a final text answer"
