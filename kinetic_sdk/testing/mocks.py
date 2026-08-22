"""Scriptable fakes so SDK users can test their agents without network calls.

``MockLLMClient`` replays a pre-programmed sequence of responses (or defers
to a callback), ``MockTool`` stands in for a real tool with a fixed result
or a handler function. Both implement the public SDK interfaces
(:class:`~kinetic_sdk.llm.client.LLMClient`, :class:`~kinetic_sdk.tool.base.Tool`),
so an agent built on them behaves exactly as it would against real backends.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
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

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
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


def tool_response(call_id: str, name: str, arguments: dict[str, Any]) -> LLMResponse:
    """Build a response requesting one tool call."""
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
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
