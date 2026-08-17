"""Shared test helpers: a scriptable mock LLM and a simple echo tool.

These live under ``tests/`` (not the SDK package) because they are test-only
fixtures and must not ship as part of the public API.
"""
from __future__ import annotations

from typing import Any

from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
from kinetic_sdk.tool.base import Tool, ToolResult


class MockLLM(LLMClient):
    """An :class:`LLMClient` that replays a pre-programmed sequence of turns.

    Pass a list of :class:`LLMResponse` (or callables ``(messages, tools, system)
    -> LLMResponse``) to ``responses``. Each call to :meth:`chat` pops the next
    entry. Callables let tests branch on the conversation if needed. If the
    script runs out, a final empty response is returned.
    """

    def __init__(self, responses: list[Any], model: str = "mock-model") -> None:
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
        self.calls.append({"messages": messages, "tools": tools, "system": system, "kwargs": kwargs})
        if not self._responses:
            return LLMResponse(content="", stop_reason="end_turn")
        entry = self._responses.pop(0)
        if callable(entry):
            return entry(messages, tools, system)
        return entry


def text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, stop_reason="end_turn")


def tool_response(call_id: str, name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


class EchoTool(Tool):
    """A trivial tool that echoes its input plus an optional prefix."""

    name = "echo"
    description = "Echo back the provided message, optionally with a prefix."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Text to echo."},
            "prefix": {"type": "string", "description": "Optional prefix.", "default": ""},
        },
        "required": ["message"],
    }

    def __init__(self, prefix: str = "") -> None:
        self._default_prefix = prefix

    def execute(self, message: str, prefix: str | None = None) -> ToolResult:
        p = prefix if prefix is not None else self._default_prefix
        return ToolResult(output=f"{p}{message}")


class FailingTool(Tool):
    """A tool that always raises, to exercise the agent's error handling."""

    name = "boom"
    description = "Always raises an exception."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **params: Any) -> ToolResult:
        raise RuntimeError("kaboom")
