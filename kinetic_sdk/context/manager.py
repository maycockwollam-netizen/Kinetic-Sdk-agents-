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
from typing import Any, Callable, Protocol

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
    return bool(_message_tool_result_ids(msg))


def _message_tool_use_ids(msg: Message) -> set[str]:
    """Ids of ``tool_use`` blocks in the message (empty for plain turns)."""
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    return {
        block["id"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    }


def _message_tool_result_ids(msg: Message) -> set[str]:
    """Ids referenced by ``tool_result`` blocks in the message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    return {
        block["tool_use_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    }


def _message_has_tool_content(msg: Message) -> bool:
    """True if the message carries any tool block (``tool_use`` or result).

    A message like that is unsafe to keep in isolation: providers reject a
    ``tool_use`` whose result was elided and a ``tool_result`` whose use was
    elided, so compaction must never keep such a message on its own.
    """
    return bool(_message_tool_use_ids(msg) or _message_tool_result_ids(msg))


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

    def _count_tokens(self, text: str) -> int:
        """Count tokens of one text chunk; overridable per implementation."""
        return estimate_tokens(text)

    def estimate_state_tokens(self, state: ConversationState) -> int:
        """Estimated total tokens of system prompt + all messages."""
        total = 0
        if state.system_prompt:
            total += self._count_tokens(state.system_prompt)
        for msg in state.messages:
            total += self._count_tokens(_stringify_content(msg.get("content")))
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
        token_counter: Optional callable ``str -> int`` replacing the
            heuristic entirely (e.g.
            :class:`~kinetic_sdk.context.tokens.TiktokenCounter`). When set,
            ``chars_per_token`` is unused.

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
        token_counter: Callable[[str], int] | None = None,
        max_tool_result_chars: int | None = None,
    ) -> None:
        if keep_last_tool_results < 0:
            raise ValueError("keep_last_tool_results must be >= 0")
        if not 0 < safety_threshold <= 1:
            raise ValueError("safety_threshold must be in (0, 1]")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be >= 1")
        if max_tool_result_chars is not None and max_tool_result_chars < 100:
            raise ValueError("max_tool_result_chars must be >= 100 or None")
        self.keep_last_tool_results = keep_last_tool_results
        self.safety_threshold = safety_threshold
        self.chars_per_token = chars_per_token
        self.token_counter = token_counter
        #: When set, every kept ``tool_result`` whose text exceeds this many
        #: characters is cut to head + marker + tail. This is what saves a
        #: conversation whose protected tail alone overflows the window
        #: (e.g. 5 huge build logs) - eliding messages cannot help there.
        self.max_tool_result_chars = max_tool_result_chars

    def _count_tokens(self, text: str) -> int:
        if self.token_counter is not None:
            return self.token_counter(text)
        return estimate_tokens(text, self.chars_per_token)

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

        Tool-block integrity is guaranteed: the tail is widened so every
        kept ``tool_result`` keeps its ``tool_use`` (parallel tool calls are
        never split across the elision boundary), and the head is only kept
        when it carries no tool blocks. Providers reject histories with
        dangling tool references, so violating this would fail the next LLM
        call outright.
        """
        messages = state.messages
        if len(messages) <= 2:
            return self._copy(state, self._truncate_oversized_tool_results(list(messages)))

        tail_start = self._tail_start(messages)
        head_end = (
            1
            if tail_start > 0 and not _message_has_tool_content(messages[0])
            else 0
        )
        removed = tail_start - head_end
        if removed <= 0:
            # No message-level cut is possible, but oversized tool results in
            # the kept span may still need trimming (a giant protected tail
            # is exactly the case elision cannot fix).
            return self._copy(state, self._truncate_oversized_tool_results(list(messages)))

        kept = [dict(m) for m in messages[:head_end]]
        kept.append(self._elided_message(messages[head_end:tail_start], removed))
        kept.extend(dict(m) for m in messages[tail_start:])
        return self._copy(state, self._truncate_oversized_tool_results(kept))

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
        whole conversation when compaction is actually needed, then widened
        to a parallel-group boundary by :meth:`_expand_to_group_boundary`.
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
        start = max(0, min(anchor - 1, n - 1))
        return self._expand_to_group_boundary(messages, start)

    @staticmethod
    def _expand_to_group_boundary(messages: list[Message], start: int) -> int:
        """Move *start* back until no kept ``tool_result`` is orphaned.

        Parallel tool calls are stored as ONE assistant message carrying N
        ``tool_use`` blocks followed by N separate ``tool_result`` messages.
        A naive anchor can therefore land mid-group — keeping results whose
        ``tool_use`` was elided, which every provider rejects with a 400.
        We walk the boundary back until every ``tool_result`` in the tail
        has its ``tool_use`` inside the tail. If the references can never be
        resolved (e.g. a history already corrupted by an older buggy
        compaction), *start* reaches 0 and :meth:`compact` returns the state
        unchanged — degrading to no compaction instead of emitting an
        invalid history.
        """
        needed: set[str] = set()
        provided: set[str] = set()
        for msg in messages[start:]:
            needed |= _message_tool_result_ids(msg)
            provided |= _message_tool_use_ids(msg)
        while start > 0 and not needed <= provided:
            start -= 1
            needed |= _message_tool_result_ids(messages[start])
            provided |= _message_tool_use_ids(messages[start])
        return start

    TRUNCATION_MARKER = "\n[... {n} ký tự ở giữa đã được cắt bớt ...]\n"

    def _truncate_oversized_tool_results(self, messages: list[Message]) -> list[Message]:
        """Cut oversized ``tool_result`` text blocks to head + marker + tail.

        No-op when ``max_tool_result_chars`` is unset. Original messages are
        never mutated: messages needing a cut get a fresh dict + content list.
        """
        if self.max_tool_result_chars is None:
            return messages
        out: list[Message] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list) or not any(
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and isinstance(b.get("content"), str)
                and len(b["content"]) > self.max_tool_result_chars
                for b in content
            ):
                out.append(msg)
                continue
            new_content = [
                self._truncate_block(b) if isinstance(b, dict) else b
                for b in content
            ]
            out.append({**msg, "content": new_content})
        return out

    def _truncate_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of a ``tool_result`` block with its text trimmed."""
        if block.get("type") != "tool_result" or not isinstance(block.get("content"), str):
            return block
        text = block["content"]
        limit = self.max_tool_result_chars
        assert limit is not None  # guarded by the caller
        if len(text) <= limit:
            return block
        head = limit * 2 // 3
        tail = limit - head
        omitted = len(text) - head - tail
        return {
            **block,
            "content": text[:head] + self.TRUNCATION_MARKER.format(n=omitted) + text[-tail:],
        }

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
        token_counter: Callable[[str], int] | None = None,
        max_tool_result_chars: int | None = None,
    ) -> None:
        super().__init__(
            keep_last_tool_results=keep_last_tool_results,
            safety_threshold=safety_threshold,
            chars_per_token=chars_per_token,
            token_counter=token_counter,
            max_tool_result_chars=max_tool_result_chars,
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
