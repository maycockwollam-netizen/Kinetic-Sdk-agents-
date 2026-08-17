"""Context-window manager (Stage 2 - interface stub).

Responsible for keeping the conversation history within the model's context
window: truncating/summarising older turns while preserving the system prompt
and the most recent tool results. Stage 1 only defines the interface so the
agent can declare a dependency on it; the agent loop itself does not call into
it yet. A real implementation lands in Stage 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kinetic_sdk.conversation.state import ConversationState


class ContextManager(ABC):
    """Interface for context-window management strategies.

    Implementations mutate the provided :class:`ConversationState` in place
    (e.g. drop old messages or replace them with a summary) so the caller can
    continue using the same state object.
    """

    @abstractmethod
    def manage(self, state: ConversationState) -> None:
        """Ensure *state* fits within the configured context budget.

        Called by the agent loop before each LLM call in Stage 2. Must never
        remove the system prompt or the most recent tool result pair, per the
        KINETIC context policy.
        """


class NoopContextManager(ContextManager):
    """Stage 1 default: does nothing.

    The Stage 1 loop relies on :class:`ConversationState.max_messages` for a
    crude cap. Real summarisation/truncation is implemented in Stage 2 and
    swapped in here.
    """

    def manage(self, state: ConversationState) -> None:
        """No-op; returns without modifying *state*."""
        return None
