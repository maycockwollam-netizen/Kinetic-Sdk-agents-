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
is implemented by :class:`SummarizingContextManager` when a summarizer is
provided, with safe fallback to truncation on any summarizer failure.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.llm.client import LLMClient, Message
from kinetic_sdk.security.redact import redact_secrets, redact_value

logger = logging.getLogger(__name__)


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


class ContextSummarizer(Protocol):
    """Summarizes an elided conversation span for context compaction.

    Implementations are intentionally tiny and injectable so tests can use a
    deterministic fake while SDK users can pass an LLM-backed summarizer. They
    should return a concise human-readable summary. Empty strings are treated as
    failure by :class:`SummarizingContextManager` and trigger truncation fallback.
    """

    def summarize(self, messages: list[Message]) -> str:
        """Return a short summary of *messages*."""
        ...


class LLMContextSummarizer:
    """LLM-backed summarizer for elided conversation spans.

    The caller injects an :class:`~kinetic_sdk.llm.client.LLMClient`, keeping
    this module provider-neutral and avoiding any hard dependency on ``litellm``.
    The public ``alias`` mirrors the classifier pattern: logs/config can refer
    to the summarizer by a stable SDK alias instead of leaking a concrete model
    name.
    """

    alias = "kinetic-context-summarizer-v1"

    def __init__(self, client: LLMClient, max_tokens: int = 150) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.client = client
        self.max_tokens = max_tokens

    def summarize(self, messages: list[Message]) -> str:
        """Ask the injected LLM for a 1-2 sentence Vietnamese summary."""
        instructions = (
            "Tóm tắt phần hội thoại đã bị rút gọn trong 1-2 câu tiếng Việt. "
            "Giữ lại mục tiêu, quyết định quan trọng, lỗi/tool result đáng chú ý, "
            "và thông tin mà agent cần để tiếp tục. Không thêm thông tin mới."
        )
        body = json.dumps(messages, ensure_ascii=False, default=str)
        response = self.client.chat(
            messages=[{"role": "user", "content": f"Đoạn hội thoại cần tóm tắt:\n{body}"}],
            system=instructions,
            max_tokens=self.max_tokens,
        )
        return (response.content or "").strip()


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
        kept.append(self._elided_message(messages[head_end:tail_start], removed))
        kept.extend(dict(m) for m in messages[tail_start:])
        return self._copy(state, kept)

    def _elided_message(self, elided: list[Message], removed: int) -> Message:
        """The single message replacing the dropped middle span.

        Subclasses override this to carry richer content (e.g. a summary).
        """
        return {
            "role": "user",
            "content": self.PLACEHOLDER_TEMPLATE.format(n=removed),
        }

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
    """Summarization-based compaction with safe truncation fallback.

    When a summarizer is configured, the dropped middle span is replaced by a
    concise summary instead of a plain placeholder. Pass either a ready-made
    :class:`ContextSummarizer` (``summarizer``) or a plain
    :class:`~kinetic_sdk.llm.client.LLMClient` (``summarizer_client``, wrapped
    in :class:`LLMContextSummarizer` with a low ``summary_max_tokens`` cap) so
    SDK users can pick a cheap/fast model for summaries, mirroring the
    classifier pattern. Passing both raises ``ValueError``.

    Safety rails, in line with the rest of the SDK:

    * The elided span is scrubbed with
      :func:`~kinetic_sdk.security.redact.redact_value` before it leaves the
      process — tool results may carry credentials and the summarizer is a
      separate model call that does not need them.
    * Any summarizer failure (exception, non-string or empty summary) falls
      back to :class:`SimpleTruncateContextManager`'s static placeholder
      instead of crashing ``compact()``, and emits
      ``context.summarization_failed`` on ``event_bus`` (when configured) so
      the failure can be traced via the observability module.
    """

    SUMMARY_TEMPLATE = "[{n} tin nhắn trước đó đã được tóm tắt: {summary}]"

    #: Event emitted when summarization fails and truncation fallback kicks in.
    FAILURE_EVENT = "context.summarization_failed"

    def __init__(
        self,
        keep_last_tool_results: int = 5,
        safety_threshold: float = 0.8,
        chars_per_token: int = 4,
        summarizer: ContextSummarizer | None = None,
        summarizer_client: LLMClient | None = None,
        event_bus: EventBus | None = None,
        max_summary_chars: int = 1_000,
        summary_max_tokens: int = 150,
    ) -> None:
        super().__init__(
            keep_last_tool_results=keep_last_tool_results,
            safety_threshold=safety_threshold,
            chars_per_token=chars_per_token,
        )
        if summarizer is not None and summarizer_client is not None:
            raise ValueError("pass either `summarizer` or `summarizer_client`, not both")
        if summarizer is None and summarizer_client is not None:
            summarizer = LLMContextSummarizer(summarizer_client, max_tokens=summary_max_tokens)
        if max_summary_chars < 1:
            raise ValueError("max_summary_chars must be >= 1")
        self.summarizer = summarizer
        self.event_bus = event_bus
        self.max_summary_chars = max_summary_chars

    def _elided_message(self, elided: list[Message], removed: int) -> Message:
        summary = self._summarize(elided)
        if not summary:
            return super()._elided_message(elided, removed)
        return {
            "role": "user",
            "content": self.SUMMARY_TEMPLATE.format(n=removed, summary=summary),
        }

    def _summarize(self, messages: list[Message]) -> str:
        if self.summarizer is None:
            return ""
        # Scrub credentials out of the span before it is sent to another model.
        redacted = [redact_value(dict(m)) for m in messages]
        try:
            summary = self.summarizer.summarize(redacted)
        except Exception as exc:  # noqa: BLE001 - compaction must safely degrade
            logger.warning("Context summarization failed: %s", exc)
            self._emit_failure("exception", messages, exc)
            return ""
        if not isinstance(summary, str):
            self._emit_failure("non_string_summary", messages, None)
            return ""
        summary = " ".join(summary.split())
        if not summary:
            self._emit_failure("empty_summary", messages, None)
            return ""
        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars].rstrip() + "…"
        return summary

    def _emit_failure(
        self, reason: str, messages: list[Message], exc: Exception | None
    ) -> None:
        """Publish ``context.summarization_failed`` if a bus is attached."""
        if self.event_bus is None:
            return
        payload: dict[str, Any] = {
            "manager": type(self).__name__,
            "reason": reason,
            "elided_messages": len(messages),
        }
        if exc is not None:
            payload["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")
        self.event_bus.publish(Event(type=self.FAILURE_EVENT, payload=payload, source="context"))
