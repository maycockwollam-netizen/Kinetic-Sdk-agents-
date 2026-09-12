"""Async mirror tests for the MCP client (offline thread bridge)."""
from __future__ import annotations

import pytest

from kinetic_sdk.mcp import AsyncMCPClient, MCPClientError
from kinetic_sdk.mcp.protocol import JsonRpcNotification, JsonRpcRequest

from .test_mcp_client import FakeTransport, ok_initialize_responder

pytestmark = pytest.mark.asyncio


async def test_async_handshake_and_tools_match_sync_order() -> None:
    transport = FakeTransport(ok_initialize_responder)
    client = AsyncMCPClient(transport)
    assert not client.initialized
    await client.initialize()
    assert (await client.list_tools())[0]["name"] == "search"
    assert [message.method for message in transport.sent if isinstance(message, (JsonRpcRequest, JsonRpcNotification))] == [
        "initialize", "notifications/initialized", "tools/list"
    ]


async def test_async_call_before_initialize_raises_same_error() -> None:
    with pytest.raises(MCPClientError, match="before initialize"):
        await AsyncMCPClient(FakeTransport()).call_tool("x", {})


async def test_async_timeout_and_close() -> None:
    transport = FakeTransport()
    client = AsyncMCPClient(transport)
    with pytest.raises(Exception, match="inbox empty"):
        await client.initialize(timeout=0.001)
    await client.close()
    assert transport.closed
