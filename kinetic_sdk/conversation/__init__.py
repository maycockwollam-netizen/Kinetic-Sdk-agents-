"""Conversation package: state management for chat sessions."""

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import (
    ConversationStore,
    ConversationStoreError,
    JsonFileConversationStore,
    state_from_dict,
    state_to_dict,
)

__all__ = [
    "ConversationState",
    "ConversationStore",
    "ConversationStoreError",
    "JsonFileConversationStore",
    "state_from_dict",
    "state_to_dict",
]
