"""Async mirror tests for the MCP client (offline thread bridge)."""
from __future__ import annotations

import asyncio
import queue
import time

import pytest

from kinetic_sdk.mcp import AsyncMCPClient, MCPClientError
from kinetic_sdk.mcp.protocol import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)
from kinetic_sdk.mcp.transport import MCPTimeoutError, Transport

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


async def test_async_concurrent_tool_calls_are_serialized() -> None:
    class DelayedToolTransport(Transport):
        def __init__(self) -> None:
            self._inbox: queue.Queue[object] = queue.Queue()
            self.events: list[tuple[str, int]] = []

        def send(self, message: object) -> None:
            assert isinstance(message, (JsonRpcRequest, JsonRpcNotification))
            if isinstance(message, JsonRpcRequest) and message.method == "initialize":
                self._inbox.put(JsonRpcResponse(id=message.id, result={
                    "protocolVersion": "2025-03-26", "capabilities": {},
                }))
            elif isinstance(message, JsonRpcRequest) and message.method == "tools/call":
                self.events.append(("sent", message.id))
                # Yield the worker thread before the matching response exists.
                time.sleep(0.01)
                self.events.append(("responded", message.id))
                self._inbox.put(JsonRpcResponse(id=message.id, result={
                    "content": [{"type": "text", "text": message.params["arguments"]["value"]}],
                    "isError": False,
                }))

        def receive(self, timeout: float | None = None) -> object:
            try:
                return self._inbox.get(timeout=timeout)
            except queue.Empty:
                raise MCPTimeoutError("test inbox empty") from None

        def close(self) -> None:
            pass

    transport = DelayedToolTransport()
    client = AsyncMCPClient(transport)
    await client.initialize()
    results = await asyncio.gather(*[
        client.call_tool("echo", {"value": str(value)}) for value in range(4)
    ])
    assert [result["content"][0]["text"] for result in results] == ["0", "1", "2", "3"]
    assert transport.events == [
        (event, request_id)
        for request_id in range(2, 6)
        for event in ("sent", "responded")
    ]
