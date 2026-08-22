"""Shared test helpers: the public mock LLM re-exported plus extra fakes.

``MockLLM``/``text_response``/``tool_response`` are aliases of the public
utilities in :mod:`kinetic_sdk.testing` (kept under the old names so existing
tests stay unchanged). ``EchoTool``/``FailingTool`` are fixtures specific to
this suite and remain test-only.
"""
from __future__ import annotations

from typing import Any

from kinetic_sdk.testing.mocks import (
    MockLLMClient,
    text_response,
    tool_response,
)
from kinetic_sdk.tool.base import Tool, ToolResult

#: Backwards-compatible alias: the SDK's own tests predate ``testing/``.
MockLLM = MockLLMClient


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
