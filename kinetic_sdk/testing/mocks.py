"""Scriptable fakes so SDK users can test their agents without network calls.

``MockLLMClient`` replays a pre-programmed sequence of responses (or defers
to a callback), ``MockTool`` stands in for a real tool with a fixed result
or a handler function. Both implement the public SDK interfaces
(:class:`~kinetic_sdk.llm.client.LLMClient`, :class:`~kinetic_sdk.tool.base.Tool`),
so an agent built on them behaves exactly as it would against real backends.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterator, Sequence

from kinetic_sdk.llm.client import LLMClient, LLMResponse, StreamEvent, ToolCall
from kinetic_sdk.tool.base import Tool, ToolResult

#: A scripted response entry: a ready-made LLMResponse, or a callable
#: ``(messages, tools, system) -> LLMResponse`` for branching on the input.
ResponseEntry = LLMResponse | Callable[..., LLMResponse]


class MockLLMClient(LLMClient):
    """An :class:`LLMClient` replaying a pre-programmed sequence of turns.

    Args:
        responses: Entries consumed in order, one per :meth:`chat` call. An
            entry is either an :class:`LLMResponse` returned as-is, or a
            callable ``(messages, tools, system) -> LLMResponse`` for tests
            that branch on the conversation. When the script runs out, a
            final empty ``end_turn`` response is returned (the loop exits).
        model: Value reported on the ``model`` attribute.

    Every call is recorded on :attr:`calls` for later assertions.
    """

    def __init__(
        self, responses: Sequence[ResponseEntry], model: str = "mock-model"
    ) -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    #: Chunk size used by :meth:`chat_stream` when splitting a scripted
    #: response's content into text deltas (small on purpose so tests observe
    #: multiple deltas, like a real provider stream).
    STREAM_CHUNK_SIZE = 8

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self._next_response(messages, tools, system, kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Stream the next scripted response: text deltas, then ``done``.

        Consumes the script exactly like :meth:`chat` (one entry per turn),
        so the same script drives streaming and non-streaming agents. The
        final ``done`` event carries the full scripted response, tool calls
        included.
        """
        response = self._next_response(messages, tools, system, kwargs)
        content = response.content
        for i in range(0, len(content), self.STREAM_CHUNK_SIZE):
            yield StreamEvent(type="text", delta=content[i : i + self.STREAM_CHUNK_SIZE])
        yield StreamEvent(type="done", delta=response)

    def _next_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "tools": tools, "system": system, "kwargs": kwargs}
        )
        if not self._responses:
            return LLMResponse(content="", stop_reason="end_turn")
        entry = self._responses.pop(0)
        if callable(entry):
            return entry(messages, tools, system)
        return entry


def text_response(text: str) -> LLMResponse:
    """Build a plain final-answer response (no tool calls)."""
    return LLMResponse(content=text, stop_reason="end_turn")


def tool_response(
    call_id: str | None = None,
    name: str = "",
    arguments: dict[str, Any] | None = None,
) -> LLMResponse:
    """Build a response requesting one tool call.

    ``call_id`` is optional: when omitted (or ``None``) a random UUID-based id
    is generated, so tests that don't care about the id can write
    ``tool_response(name="calc", arguments={...})``. The historical positional
    order ``tool_response("call-1", "calc", {...})`` keeps working.
    """
    if not name:
        raise ValueError("tool_response requires a tool name")
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id or f"call-{uuid.uuid4()}",
                name=name,
                arguments=arguments if arguments is not None else {},
            )
        ],
        stop_reason="tool_use",
    )


class MockTool(Tool):
    """A configurable stand-in :class:`Tool`.

    Args:
        name: Tool name reported to the model (unique per agent).
        description: Human-readable description for the schema.
        parameters: JSON Schema of the parameters object.
        result: Fixed outcome — a :class:`ToolResult` returned verbatim (use
            ``ToolResult(error=...)`` to exercise error/escalation paths) or
            any other value wrapped in ``ToolResult(output=value)``.
        handler: Callable ``(**params) -> ToolResult | Any``; takes precedence
            over nothing but may not be combined with ``result``.

    Every invocation is recorded on :attr:`calls`.
    """

    def __init__(
        self,
        name: str = "mock_tool",
        *,
        description: str = "A mock tool for tests.",
        parameters: dict[str, Any] | None = None,
        result: Any = None,
        handler: Callable[..., Any] | None = None,
    ) -> None:
        if result is not None and handler is not None:
            raise ValueError("Pass either a fixed result or a handler, not both.")
        self.name = name
        self.description = description
        self.parameters = parameters if parameters is not None else {
            "type": "object",
            "properties": {},
        }
        self._result = result
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def execute(self, **params: Any) -> ToolResult:
        self.calls.append(params)
        if self._handler is not None:
            return self._as_result(self._handler(**params))
        if self._result is not None:
            return self._as_result(self._result)
        return ToolResult(output={"tool": self.name, "params": params})

    @staticmethod
    def _as_result(value: Any) -> ToolResult:
        return value if isinstance(value, ToolResult) else ToolResult(output=value)
