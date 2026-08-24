"""Advanced checkpointing: fork a conversation branch, rewind a turn.

Plain persistence (``conversation/store.py``) is linear: one conversation,
one save slot. LangGraph-style "durable execution" needs two more
operations users keep asking for:

* :func:`fork_store` — copy the saved conversation into a NEW store so two
  branches (e.g. "try plan A" / "try plan B") proceed independently.
* :func:`rewind_state` — truncate the tail messages to return to an
  earlier point. The cut must land on a replay-valid boundary (never on an
  assistant message carrying ``tool_use`` without its results — providers
  reject dangling tool calls, so the raise is loud rather than corrupting
  the history).
"""

from __future__ import annotations

from pathlib import Path

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import ConversationStore, JsonFileConversationStore


class CheckpointError(Exception):
    """A checkpoint operation cannot preserve replay validity."""


def fork_store(
    source: ConversationStore, destination_path: str | Path
) -> JsonFileConversationStore:
    """Copy the saved conversation of *source* into a fresh file store.

    The destination must not exist yet (never silently overwrite a branch).
    When the source has no saved state, the destination is created empty
    (load() -> None), which still counts as a valid branch start.
    """
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise CheckpointError(
            f"fork destination {destination_path} already exists (refusing to overwrite)"
        )
    destination = JsonFileConversationStore(destination_path)
    state = source.load()
    if state is not None:
        destination.save(state)
    return destination


def rewind_state(state: ConversationState, keep_messages: int) -> ConversationState:
    """Return a NEW state with only the first *keep_messages* messages.

    The cut is rejected when the new tail would be an assistant message
    containing ``tool_use`` blocks — a dangling tool request would make the
    very next replay fail at the provider. Everything else (user turns,
    assistant text, tool_result messages) is a valid boundary.
    """
    if not isinstance(keep_messages, int) or keep_messages < 0:
        raise CheckpointError(f"keep_messages must be a non-negative int, got {keep_messages!r}")
    messages = state.messages[:keep_messages]
    if messages:
        tail = messages[-1]
        blocks = tail.get("content")
        if tail.get("role") == "assistant" and isinstance(blocks, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks):
                raise CheckpointError(
                    "rewind cut lands on a dangling tool_use (history would not "
                    "replay); cut one message earlier/later"
                )
    return ConversationState(
        system_prompt=state.system_prompt,
        messages=list(messages),
        max_messages=state.max_messages,
        metadata=dict(state.metadata),
    )
