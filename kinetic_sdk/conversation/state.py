"""In-memory conversation state for the agent loop.

Stage 1 stores the message history in memory only. Persistence to disk is a
Stage 2+ concern; the public API here (``add_user_message``, ``add_assistant``,
``add_tool_result``) is designed so a persistent backend can be swapped in
later without changing the agent code.

The message format matches Anthropic's Messages API: each message is a dict
with ``role`` and ``content`` (a string for plain text turns, or a list of
typed content blocks for assistant turns with tool use and tool results).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kinetic_sdk.llm.client import Message


@dataclass
class ConversationState:
    """Mutable conversation history + bookkeeping.

    Attributes:
        system_prompt: The system prompt sent to the model. Stored separately
            from the message list because Anthropic expects it outside of
            ``messages``. Set once at creation; changing it mid-conversation
            is discouraged.
        messages: Ordered list of messages (user / assistant / tool turns).
        max_messages: Soft cap on history length. When exceeded the oldest
            non-system messages are dropped via :meth:`truncate`. ``None``
            means unbounded. Full context-window-aware truncation/summarisation
            lives in :mod:`kinetic_sdk.context` (Stage 2).
        metadata: Free-form bag for the application (session id, user id, ...).
    """

    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    max_messages: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- mutation -----------------------------------------------------

    def add_user_message(self, content: str) -> Message:
        """Append a user turn and return the new message dict."""
        msg: Message = {"role": "user", "content": content}
        self.messages.append(msg)
        self._enforce_cap()
        return msg

    def add_assistant(self, content: Message) -> Message:
        """Append an assistant turn from a provider response.

        *content* is the raw ``content`` block(s) returned by the LLM
        (a string or a list of typed blocks including tool_use). Storing the
        raw shape keeps round-trips to the provider faithful.
        """
        msg: Message = {"role": "assistant", "content": content}
        self.messages.append(msg)
        self._enforce_cap()
        return msg

    def add_assistant_text(self, text: str) -> Message:
        """Convenience helper for an assistant turn containing only text."""
        return self.add_assistant(text)

    def add_tool_result(
        self, tool_call_id: str, output: Any, is_error: bool = False
    ) -> Message:
        """Append a ``tool_result`` turn answering a tool call.

        Anthropic models expect tool results as a separate ``user`` turn whose
        content is a list of ``tool_result`` blocks. We follow that shape so
        the history can be sent back to the model verbatim.
        """
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": output if isinstance(output, str) else str(output),
        }
        if is_error:
            block["is_error"] = True
        msg: Message = {"role": "user", "content": [block]}
        self.messages.append(msg)
        self._enforce_cap()
        return msg

    def _enforce_cap(self) -> None:
        """Drop oldest messages when ``max_messages`` is exceeded.

        We never drop the very first user turn if possible, to preserve the
        original task; this is a simple heuristic. Smarter truncation is the
        job of :mod:`kinetic_sdk.context` (Stage 2).
        """
        if self.max_messages is None:
            return
        while len(self.messages) > self.max_messages and len(self.messages) > 1:
            self.messages.pop(0)

    # --- read access --------------------------------------------------

    def for_llm(self) -> tuple[str | None, list[Message]]:
        """Return ``(system_prompt, messages)`` ready to send to an LLM."""
        return self.system_prompt, list(self.messages)

    def reset(self) -> None:
        """Clear the message history (keeps the system prompt)."""
        self.messages.clear()

    @property
    def length(self) -> int:
        """Number of messages currently stored (excluding system prompt)."""
        return len(self.messages)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length
