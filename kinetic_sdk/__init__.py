"""Kinetic Agent SDK.

A modular SDK for building AI coding agents. It includes the core agent loop,
context and memory, security and observability, extensions, and the Stage 5
production-readiness features described in the changelog.

The most commonly used names are re-exported here so newcomers can write
``from kinetic_sdk import Agent`` instead of reaching into submodules. The
list is deliberately short — everything else is imported from its submodule
(e.g. ``kinetic_sdk.testing``, ``kinetic_sdk.observability``).
"""

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LLMClient,
    LLMResponse,
    StreamEvent,
    ToolCall,
)
from kinetic_sdk.security.policy import (
    AllowListPolicy,
    PermissionDecision,
    PermissionPolicy,
    PermissivePolicy,
)
from kinetic_sdk.tool.base import Tool, ToolResult

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentMode",
    "AllowListPolicy",
    "AsyncAgent",
    "AsyncLLMClient",
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
    "__version__",
]
