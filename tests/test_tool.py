"""Unit tests for the abstract Tool interface and ToolResult."""

from __future__ import annotations

import pytest

from kinetic_sdk.tool.base import Tool, ToolResult
from tests._helpers import EchoTool


def test_tool_result_success_defaults():
    result = ToolResult(output="hello")
    assert result.output == "hello"
    assert result.error is None
    assert result.is_error is False
    assert result.metadata == {}


def test_tool_result_error_flag():
    err = ToolResult(error="boom")
    assert err.is_error is True
    assert err.output is None


def test_tool_result_empty_string_error_is_not_error():
    # An empty error string is treated as "no error" so callers can safely
    # pass through default values.
    result = ToolResult(output="ok", error="")
    assert result.is_error is False


def test_tool_result_metadata_is_unique_per_instance():
    a = ToolResult()
    b = ToolResult()
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_tool_is_abstract():
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_echo_tool_execute():
    tool = EchoTool()
    result = tool.execute(message="world")
    assert result.output == "world"
    assert result.is_error is False


def test_to_schema_shape():
    tool = EchoTool(prefix=">> ")
    schema = tool.to_schema()
    assert schema["name"] == "echo"
    assert schema["description"]
    assert schema["input_schema"]["properties"]["message"]["type"] == "string"
    assert "message" in schema["input_schema"]["required"]


def test_each_tool_instance_has_own_state():
    # Tool subclasses use ClassVar for metadata; per-instance state must be
    # isolated. EchoTool stores its prefix on the instance.
    a = EchoTool(prefix="A:")
    b = EchoTool(prefix="B:")
    assert a.execute(message="x").output == "A:x"
    assert b.execute(message="x").output == "B:x"
