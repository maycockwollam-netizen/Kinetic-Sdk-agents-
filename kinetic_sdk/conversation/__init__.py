"""Conversation package: state management for chat sessions."""

from kinetic_sdk.conversation.checkpoint import (
    CheckpointError,
    fork_store,
    rewind_state,
)
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import (
    ConversationStore,
    ConversationStoreError,
    JsonFileConversationStore,
    state_from_dict,
    state_to_dict,
)

__all__ = [
    "CheckpointError",
    "ConversationState",
    "ConversationStore",
    "ConversationStoreError",
    "JsonFileConversationStore",
    "state_from_dict",
    "state_to_dict",
    "fork_store",
    "rewind_state",
]
