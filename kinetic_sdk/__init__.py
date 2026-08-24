"""Kinetic Agent SDK.

A modular SDK for building AI coding agents. Stage 1 provides the core
tool-calling loop: ``agent``, ``conversation``, ``event``, ``llm`` and ``tool``
together with light stubs for later stages.

The most commonly used names are re-exported here so newcomers can write
``from kinetic_sdk import Agent`` instead of reaching into submodules. The
list is deliberately short — everything else is imported from its submodule
(e.g. ``kinetic_sdk.testing``, ``kinetic_sdk.observability``).
"""

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.llm.client import (
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

__version__ = "0.1.0"

__all__ = [
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
    "__version__",
]
