"""Async MCP client backed by the existing synchronous transports.

The settled Stage 6 choice is ``asyncio.to_thread`` rather than new native
async transports: ``Transport`` is blocking today and the SDK's default
async tool path uses the same bridge.  Delegating to :class:`MCPClient`
keeps wire shapes, correlation and exceptions exactly identical without
duplicating protocol logic.  Native transports can override this later when
there is a concrete scalability need.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from kinetic_sdk.mcp.client import MCPClient
from kinetic_sdk.mcp.protocol import JsonRpcNotification
from kinetic_sdk.mcp.transport import Transport


class AsyncMCPClient:
    """Awaitable façade over :class:`MCPClient` with identical semantics.

    One instance serializes concurrent callers: each request/response cycle
    queues behind the previous one because the synchronous client correlates
    replies by reading the next matching message. Callers needing true
    parallel MCP calls must use separate clients and transports (or server
    subprocesses), rather than sharing one instance.
    """

    def __init__(self, transport: Transport, init_timeout: float = 10.0,
                 request_timeout: float = 30.0, client_name: str = "kinetic-agent-sdk",
                 client_version: str = "0.1.0",
                 on_notification: Callable[[JsonRpcNotification], None] | None = None) -> None:
        self._sync = MCPClient(transport, init_timeout, request_timeout, client_name,
                               client_version, on_notification)
        self._lock = asyncio.Lock()

    @property
    def initialized(self) -> bool:
        return self._sync.initialized

    @property
    def protocol_version(self) -> str | None:
        return self._sync.protocol_version

    @property
    def server_info(self) -> dict[str, Any] | None:
        return self._sync.server_info

    async def initialize(self, timeout: float | None = None) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._sync.initialize, timeout)

    async def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._sync.list_tools, timeout)

    async def call_tool(self, name: str, arguments: dict[str, Any],
                        timeout: float | None = None) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._sync.call_tool, name, arguments, timeout)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync.close)

    async def __aenter__(self) -> "AsyncMCPClient": return self
    async def __aexit__(self, *exc_info: object) -> None: await self.close()
