"""The package root re-exports the names newcomers need most."""

from __future__ import annotations

import kinetic_sdk


def test_top_level_reexports():
    expected = {
        "Agent",
        "AgentMode",
        "AllowListPolicy",
        "ConversationState",
        "Event",
        "EventBus",
        "LLMClient",
        "LLMResponse",
        "PermissionDecision",
        "PermissionPolicy",
        "PermissivePolicy",
        "StreamEvent",
        "Tool",
        "ToolCall",
        "ToolResult",
    }
    for name in expected:
        assert hasattr(kinetic_sdk, name), f"kinetic_sdk.{name} missing"
    assert expected <= set(kinetic_sdk.__all__)


def test_agent_constructible_from_top_level_imports():
    from kinetic_sdk import Agent, PermissivePolicy
    from kinetic_sdk.testing import MockLLMClient, text_response

    agent = Agent(
        llm=MockLLMClient([text_response("ok")]),
        permission_policy=PermissivePolicy(),
    )
    assert agent.run("hi") == "ok"
