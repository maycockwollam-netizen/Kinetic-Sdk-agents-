"""Provider-agnostic LLM client interface with an Anthropic implementation.

The agent loop talks to :class:`LLMClient` only. Stage 1 ships an Anthropic
implementation (:class:`AnthropicClient`) plus a synchronous stub interface
(:class:`LLMClient`) so the agent can be tested without network access.

Design notes:
* ``LLMClient.chat`` takes a list of messages and a list of tool definitions
  and returns an :class:`LLMResponse` describing the model's turn: text and
  any tool calls it wants the agent to run.
* Streaming is exposed via ``chat_stream`` which yields incremental
  :class:`StreamEvent` deltas. The non-streaming ``chat`` is built on top of
  ``chat_stream`` for the Anthropic backend so there is one code path to
  maintain. A non-streaming provider can implement ``chat`` directly and
  leave ``chat_stream`` raising ``NotImplementedError``.
* ``anthropic`` is an optional dependency: import this module without it
  installed only fails when you actually instantiate :class:`AnthropicClient`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal

#: Role of a message in the conversation.
Role = Literal["system", "user", "assistant", "tool"]

#: A single message in the conversation history. The shape intentionally
#: matches Anthropic's message format (``role`` + ``content`` where ``content``
#: is a list of typed blocks) because that is the most expressive of the
#: common provider formats. Other providers translate to/from it.
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    Attributes:
        id: Provider-assigned id used to correlate the eventual tool result
            with this call.
        name: Name of the tool to invoke (must match a registered Tool).
        arguments: Parsed arguments object. Parsed from JSON where the
            provider sends a string.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """The model's turn after a ``chat`` call.

    Attributes:
        content: Full assistant text for the turn, concatenated from any
            text blocks the model produced. Empty when the turn only
            requested tool calls.
        tool_calls: Tool calls the model wants executed. Empty when the
            model is done and producing a final text answer.
        stop_reason: Provider-native stop reason (e.g. ``"end_turn"`` or
            ``"tool_use"``). Useful for debugging and tests.
        usage: Token usage info with at least ``input_tokens`` and
            ``output_tokens`` when the provider reports it.
        raw: The unprocessed provider response, for advanced use.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class StreamEvent:
    """An incremental delta from a streaming response.

    ``type`` is one of:
    - ``"text"``: a chunk of assistant text (``delta`` is the text).
    - ``"tool_call"``: a complete tool call (parsed once the provider has
      finished emitting it; streaming partial tool args is a later stage).
    - ``"done"``: the stream is complete, carries the final :class:`LLMResponse`.
    """

    type: Literal["text", "tool_call", "done"]
    delta: str | ToolCall | LLMResponse | None = None


class LLMClient(ABC):
    """Abstract interface every provider implementation must satisfy."""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a single non-streaming chat turn and return the full response.

        Args:
            messages: Conversation history (user/assistant/tool turns) in
                Anthropic message format. System prompt is passed separately.
            tools: Tool definitions (provider-neutral shape from
                :meth:`kinetic_sdk.tool.base.Tool.to_schema`) the model may
                call. ``None`` disables tool use.
            system: Optional system prompt prepended to the conversation.
            **kwargs: Provider-specific overrides (temperature, max_tokens,
                stop sequences, ...).
        """

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Yield incremental :class:`StreamEvent` deltas.

        Default implementation raises so providers that don't stream remain
        usable via :meth:`chat` without inheriting a broken streaming API.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")


class AnthropicClient(LLMClient):
    """LLM client backed by the Anthropic Messages API.

    Requires the optional ``anthropic`` package. Streaming and
    non-streaming share one code path: ``chat`` aggregates ``chat_stream``.

    The Anthropic SDK is imported lazily so importing this module (and the
    rest of the SDK) does not force the dependency on environments that use
    a different provider.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # Allow injecting a mock/real client for testing.
        self._client = client or self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: str | None) -> Any:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install it with: pip install kinetic-agent-sdk[anthropic]"
            ) from exc
        return anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat; built by aggregating :meth:`chat_stream`."""
        final: LLMResponse | None = None
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for ev in self.chat_stream(messages, tools, system, **kwargs):
            if ev.type == "text" and isinstance(ev.delta, str):
                text_parts.append(ev.delta)
            elif ev.type == "tool_call" and isinstance(ev.delta, ToolCall):
                tool_calls.append(ev.delta)
            elif ev.type == "done" and isinstance(ev.delta, LLMResponse):
                final = ev.delta
        if final is None:  # pragma: no cover - defensive
            final = LLMResponse(content="".join(text_parts), tool_calls=tool_calls)
        return final

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Stream the Anthropic Messages API as :class:`StreamEvent` deltas.

        Uses the synchronous streaming endpoint. Partial text is emitted as
        ``text`` events; tool-use blocks are parsed and emitted as a single
        ``tool_call`` event once their arguments JSON is complete. The final
        aggregated :class:`LLMResponse` is emitted in a ``done`` event.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if tools:
            request["tools"] = tools
        request.update(kwargs)

        text_buf: list[str] = []
        tool_calls: list[ToolCall] = []
        # active tool_use blocks keyed by index -> accumulating json string.
        tool_bufs: dict[int, dict[str, str]] = {}
        stop_reason: str | None = None
        usage: dict[str, int] = {}

        with self._client.messages.stream(**request) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        idx = getattr(event, "index", 0)
                        tool_bufs[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input": "",
                        }
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)
                    idx = getattr(event, "index", 0)
                    if dtype == "text_delta":
                        chunk = getattr(delta, "text", "")
                        text_buf.append(chunk)
                        yield StreamEvent(type="text", delta=chunk)
                    elif dtype == "input_json_delta":
                        chunk = getattr(delta, "partial_json", "")
                        if idx in tool_bufs:
                            tool_bufs[idx]["input"] += chunk
                elif etype == "content_block_stop":
                    idx = getattr(event, "index", 0)
                    if idx in tool_bufs:
                        tb = tool_bufs.pop(idx)
                        import json

                        try:
                            args = json.loads(tb["input"] or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw": tb["input"]}
                        call = ToolCall(id=tb["id"], name=tb["name"], arguments=args)
                        tool_calls.append(call)
                        yield StreamEvent(type="tool_call", delta=call)
                elif etype == "message_delta":
                    msg = getattr(event, "delta", None)
                    stop_reason = getattr(msg, "stop_reason", stop_reason)
                    u = getattr(event, "usage", None)
                    if u is not None:
                        usage.update(
                            {
                                "input_tokens": getattr(u, "input_tokens", usage.get("input_tokens", 0)),
                                "output_tokens": getattr(u, "output_tokens", usage.get("output_tokens", 0)),
                            }
                        )
                elif etype == "message_start":
                    msg = getattr(event, "message", None)
                    u = getattr(msg, "usage", None)
                    if u is not None:
                        usage.update(
                            {
                                "input_tokens": getattr(u, "input_tokens", 0),
                                "output_tokens": getattr(u, "output_tokens", 0),
                            }
                        )

        final = LLMResponse(
            content="".join(text_buf),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=None,
        )
        yield StreamEvent(type="done", delta=final)


class AsyncLLMClient(ABC):
    """Async variant of :class:`LLMClient` for streaming-first providers.

    Stage 1 keeps the agent loop synchronous for determinism; this interface
    is provided so future async providers can plug in without reshaping the
    agent. It mirrors :class:`LLMClient` with awaitable methods.
    """

    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming variant. Yields :class:`StreamEvent` deltas."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        # Make this a generator for type checkers even though it never runs.
        yield StreamEvent(type="done", delta=LLMResponse())  # pragma: no cover
