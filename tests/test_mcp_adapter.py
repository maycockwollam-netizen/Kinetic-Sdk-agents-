"""Tests for kinetic_sdk.mcp.adapter (MCPToolAdapter)."""

from __future__ import annotations

from typing import Any

import pytest

from kinetic_sdk.mcp.adapter import MCPToolAdapter
from kinetic_sdk.mcp.client import MCPClientError, MCPServerError
from kinetic_sdk.mcp.transport import MCPTransportError
from kinetic_sdk.tool.base import Tool


class StubMCPClient:
    """Stands in for MCPClient: records calls, replays scripted results."""

    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self.result = result if result is not None else {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        }
        self.exc = exc
        self.calls: list[tuple[str, dict, Any]] = []

    def call_tool(self, name: str, arguments: dict, timeout: float | None = None) -> Any:
        self.calls.append((name, arguments, timeout))
        if self.exc is not None:
            raise self.exc
        return self.result


ECHO_SCHEMA = {
    "name": "echo",
    "description": "Echo back text",
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
}


class TestInterface:
    def test_is_a_tool_instance(self) -> None:
        adapter = MCPToolAdapter(StubMCPClient(), "echo", "unity")
        assert isinstance(adapter, Tool)

    def test_name_is_prefixed_with_server(self) -> None:
        adapter = MCPToolAdapter(StubMCPClient(), "search", "unity")
        assert adapter.name == "unity.search"
        assert adapter.mcp_tool_name == "search"
        assert adapter.server_name == "unity"

    def test_two_servers_same_tool_name_do_not_collide(self) -> None:
        a = MCPToolAdapter(StubMCPClient(), "search", "unity")
        b = MCPToolAdapter(StubMCPClient(), "search", "roblox")
        assert a.name != b.name
        assert {a.name, b.name} == {"unity.search", "roblox.search"}

    def test_from_mcp_schema_maps_camel_case(self) -> None:
        adapter = MCPToolAdapter.from_mcp_schema(StubMCPClient(), "unity", ECHO_SCHEMA)
        assert adapter.name == "unity.echo"
        assert adapter.description == "Echo back text"
        assert adapter.parameters == ECHO_SCHEMA["inputSchema"]
        assert adapter.to_schema() == {
            "name": "unity.echo",
            "description": "Echo back text",
            "input_schema": ECHO_SCHEMA["inputSchema"],
        }

    def test_from_mcp_schema_requires_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            MCPToolAdapter.from_mcp_schema(StubMCPClient(), "unity", {"description": "x"})

    def test_defaults_for_missing_description_and_schema(self) -> None:
        adapter = MCPToolAdapter(StubMCPClient(), "ping", "local")
        assert "local" in adapter.description
        assert adapter.parameters == {"type": "object", "properties": {}}

    def test_empty_names_rejected(self) -> None:
        with pytest.raises(ValueError):
            MCPToolAdapter(StubMCPClient(), "", "unity")
        with pytest.raises(ValueError):
            MCPToolAdapter(StubMCPClient(), "echo", "")


class TestExecute:
    def test_success_maps_text_content_to_output(self) -> None:
        client = StubMCPClient(
            result={"content": [{"type": "text", "text": "found 3 objects"}], "isError": False}
        )
        adapter = MCPToolAdapter(client, "search", "unity")
        result = adapter.execute(text="cube")
        assert not result.is_error
        assert result.output == "found 3 objects"
        assert result.metadata == {"mcp_server": "unity", "mcp_tool": "search"}
        # The UNPREFIXED name goes on the wire, not "unity.search".
        assert client.calls == [("search", {"text": "cube"}, None)]

    def test_mcp_is_error_maps_to_tool_result_error(self) -> None:
        client = StubMCPClient(
            result={"content": [{"type": "text", "text": "object not found"}], "isError": True}
        )
        adapter = MCPToolAdapter(client, "search", "unity")
        result = adapter.execute(text="missing")
        assert result.is_error
        assert "object not found" in result.error

    def test_multi_block_content_stays_structured(self) -> None:
        blocks = [
            {"type": "text", "text": "part 1"},
            {"type": "text", "text": "part 2"},
        ]
        client = StubMCPClient(result={"content": blocks, "isError": False})
        adapter = MCPToolAdapter(client, "report", "unity")
        result = adapter.execute()
        assert result.output == blocks

    def test_empty_content_becomes_empty_string(self) -> None:
        client = StubMCPClient(result={"content": [], "isError": False})
        adapter = MCPToolAdapter(client, "noop", "unity")
        assert adapter.execute().output == ""

    def test_transport_error_becomes_tool_result_error_not_exception(self) -> None:
        client = StubMCPClient(exc=MCPTransportError("subprocess died; process exited with code 3"))
        adapter = MCPToolAdapter(client, "search", "unity")
        result = adapter.execute(text="x")
        assert result.is_error
        assert "subprocess died" in result.error

    def test_server_error_becomes_tool_result_error(self) -> None:
        client = StubMCPClient(exc=MCPServerError(code=-32602, message="unknown tool"))
        adapter = MCPToolAdapter(client, "nope", "unity")
        result = adapter.execute()
        assert result.is_error
        assert "unknown tool" in result.error

    def test_client_error_becomes_tool_result_error(self) -> None:
        client = StubMCPClient(exc=MCPClientError("not initialized"))
        adapter = MCPToolAdapter(client, "x", "unity")
        assert adapter.execute().is_error

    def test_malformed_result_becomes_tool_result_error(self) -> None:
        client = StubMCPClient(result="not-a-dict")
        adapter = MCPToolAdapter(client, "x", "unity")
        result = adapter.execute()
        assert result.is_error
        assert "malformed" in result.error

    def test_timeout_forwarded_to_client(self) -> None:
        client = StubMCPClient()
        adapter = MCPToolAdapter(client, "slow", "unity", timeout=7.5)
        adapter.execute()
        assert client.calls[0][2] == 7.5


class TestRedaction:
    def test_secret_in_output_is_redacted(self) -> None:
        token = "ghp_" + "a1b2c3d4e5f6" * 3
        client = StubMCPClient(
            result={"content": [{"type": "text", "text": f"token: {token}"}], "isError": False}
        )
        adapter = MCPToolAdapter(client, "leak", "untrusted")
        result = adapter.execute()
        assert token not in result.output
        assert "[REDACTED]" in result.output

    def test_secret_in_error_output_is_redacted(self) -> None:
        token = "sk-" + "z9y8x7w6v5" * 4
        client = StubMCPClient(
            result={"content": [{"type": "text", "text": f"auth failed for {token}"}], "isError": True}
        )
        adapter = MCPToolAdapter(client, "leak", "untrusted")
        result = adapter.execute()
        assert result.is_error
        assert token not in result.error

    def test_secret_in_structured_blocks_is_redacted(self) -> None:
        token = "ghp_" + "a1b2c3d4e5f6" * 3
        blocks = [
            {"type": "text", "text": "part 1"},
            {"type": "resource", "resource": {"uri": f"file:///{token}"}},
        ]
        client = StubMCPClient(result={"content": blocks, "isError": False})
        adapter = MCPToolAdapter(client, "leak", "untrusted")
        result = adapter.execute()
        assert token not in str(result.output)
