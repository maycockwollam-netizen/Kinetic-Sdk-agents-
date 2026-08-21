"""Context-window management (Stage 2 - real implementation).

Keeps the conversation history sent to the LLM within the model's context
window without losing important information. The policy, in order of what is
preserved first:

1. The system prompt (stored on the state, never touched here), the first
   user message (usually the original task), and the N most recent tool
   results (N configurable, default 5).
2. Everything in between is a candidate for compaction once the estimated
   token count crosses a safety threshold of the model's context limit.

The only compaction technique shipped complete for now is simple truncation:
the dropped middle span is replaced by a single placeholder message
(e.g. ``"[12 tin nhắn trước đó đã được rút gọn]"``). LLM-based summarisation
is exposed via :class:`SummarizingContextManager` as an extension point but
is intentionally not implemented yet (falls back to truncation).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.llm.client import Message


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Rough token estimate: ``len(text) // chars_per_token``.

    This is a crude heuristic calibrated for English text (~4 chars/token for
    common tokenizers). It tends to UNDERESTIMATE for Vietnamese and code,
    whose tokenizers split differently - treat the result as a lower-bound
    approximation and compensate via the manager's safety threshold. A
    precise ``tiktoken``-backed estimator may be added later as an optional
    dependency; the heuristic stays the zero-dependency default.
    """
    return max(1, len(text) // chars_per_token)


def _stringify_content(content: Any) -> str:
    """Flatten a message's content (str or typed blocks) to text for sizing."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _message_has_tool_result(msg: Message) -> bool:
    """True if the message carries at least one ``tool_result`` block."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


class ContextManager(ABC):
    """Interface for context-window management strategies.

    Implementations decide *when* the history is getting too big
    (:meth:`should_compact`) and produce a *new*, reduced
    :class:`ConversationState` (:meth:`compact`). Compaction is
    immutable-style: the original state object is never modified, so callers
    holding a reference to it are unaffected.
    """

    @abstractmethod
    def should_compact(self, state: ConversationState, model_context_limit: int) -> bool:
        """Decide whether *state* needs compaction.

        Compares the estimated token count of the current history against
        *model_context_limit* using a safety threshold (e.g. 80% of the
        limit) rather than waiting until the window is completely full.
        """

    @abstractmethod
    def compact(self, state: ConversationState) -> ConversationState:
        """Return a NEW, reduced copy of *state*; never mutates the original."""

    # --- shared helpers ------------------------------------------------

    def estimate_state_tokens(self, state: ConversationState) -> int:
        """Estimated total tokens of system prompt + all messages."""
        total = 0
        if state.system_prompt:
            total += estimate_tokens(state.system_prompt)
        for msg in state.messages:
            total += estimate_tokens(_stringify_content(msg.get("content")))
        return total


class NoopContextManager(ContextManager):
    """Never compacts. Useful as an opt-out or in tests."""

    def should_compact(self, state: ConversationState, model_context_limit: int) -> bool:
        """Always returns False."""
        return False

    def compact(self, state: ConversationState) -> ConversationState:
        """Returns a shallow copy of *state* unchanged."""
        return ConversationState(
            system_prompt=state.system_prompt,
            messages=[dict(m) for m in state.messages],
            max_messages=state.max_messages,
            metadata=dict(state.metadata),
        )


class SimpleTruncateContextManager(ContextManager):
    """Truncation-based compaction, the Stage 2 default.

    Args:
        keep_last_tool_results: How many of the most recent tool-result
            messages are always preserved (default 5). Their immediate
            neighbours (the assistant ``tool_use`` turn etc.) are kept too
            when they fall inside the protected tail.
        safety_threshold: Fraction of ``model_context_limit`` at which
            :meth:`should_compact` fires (default 0.8). Below 1.0 so the
            loop compacts early instead of riding the window edge.
        chars_per_token: Heuristic divisor for :func:`estimate_tokens`.

    Compaction keeps, in order: the first message (normally the original
    user request) and the tail of the conversation starting just before the
    Nth-from-last tool result. Everything in between is replaced by a single
    placeholder message so the model still sees that something was elided.
    """

    PLACEHOLDER_TEMPLATE = "[{n} tin nhắn trước đó đã được rút gọn]"

    def __init__(
        self,
        keep_last_tool_results: int = 5,
        safety_threshold: float = 0.8,
        chars_per_token: int = 4,
    ) -> None:
        if keep_last_tool_results < 0:
            raise ValueError("keep_last_tool_results must be >= 0")
        if not 0 < safety_threshold <= 1:
            raise ValueError("safety_threshold must be in (0, 1]")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be >= 1")
        self.keep_last_tool_results = keep_last_tool_results
        self.safety_threshold = safety_threshold
        self.chars_per_token = chars_per_token

    # --- ContextManager interface ---------------------------------------

    def should_compact(self, state: ConversationState, model_context_limit: int) -> bool:
        """True when the estimate crosses ``safety_threshold`` of the limit."""
        if model_context_limit <= 0:
            raise ValueError("model_context_limit must be positive")
        budget = model_context_limit * self.safety_threshold
        return self.estimate_state_tokens(state) >= budget

    def compact(self, state: ConversationState) -> ConversationState:
        """Drop the middle of the history, keeping head + recent tail.

        The tail starts at (or just before) the Nth-most-recent tool result;
        the head is the first message if it does not overlap the tail. A
        single placeholder message records how many messages were elided.
        Edge cases (0-2 messages, or everything protected) return an
        unchanged copy.
        """
        messages = state.messages
        if len(messages) <= 2:
            return self._copy(state, list(messages))

        tail_start = self._tail_start(messages)
        head_end = 1 if tail_start > 0 else 0  # keep original request iff outside the tail
        removed = tail_start - head_end
        if removed <= 0:
            return self._copy(state, list(messages))

        kept = [dict(m) for m in messages[:head_end]]
        placeholder: Message = {
            "role": "user",
            "content": self.PLACEHOLDER_TEMPLATE.format(n=removed),
        }
        kept.append(placeholder)
        kept.extend(dict(m) for m in messages[tail_start:])
        return self._copy(state, kept)

    # --- internals --------------------------------------------------------

    def _tail_start(self, messages: list[Message]) -> int:
        """Index where the protected tail begins.

        One message before the Nth-from-last tool result (to also keep the
        assistant turn that requested it), clamped so the tail is never the
        whole conversation when compaction is actually needed.
        """
        n = len(messages)
        if self.keep_last_tool_results == 0:
            return n  # nothing tool-specific to protect; head+placeholder only
        tool_result_idx = [
            i for i, msg in enumerate(messages) if _message_has_tool_result(msg)
        ]
        if len(tool_result_idx) >= self.keep_last_tool_results:
            anchor = tool_result_idx[-self.keep_last_tool_results]
        elif tool_result_idx:
            anchor = tool_result_idx[0]
        else:
            anchor = n - 1
        return max(0, min(anchor - 1, n - 1))

    @staticmethod
    def _copy(state: ConversationState, messages: list[Message]) -> ConversationState:
        """Build a new state with fresh list/dict containers (immutable-style)."""
        return ConversationState(
            system_prompt=state.system_prompt,
            messages=messages,
            max_messages=state.max_messages,
            metadata=dict(state.metadata),
        )


class SummarizingContextManager(SimpleTruncateContextManager):
    """Extension point: LLM-summarised compaction. NOT IMPLEMENTED yet.

    The plan is to replace the dropped middle span with a 1-2 sentence
    summary produced by a cheap model (behind an alias, like the classifier)
    instead of the plain placeholder, so the agent retains a sketch of the
    elided context. Until that lands, this class behaves exactly like
    :class:`SimpleTruncateContextManager`.
    """
