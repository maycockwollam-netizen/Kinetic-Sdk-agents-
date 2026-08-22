"""MCP client (Stage 4, MCP module).

:class:`MCPClient` talks to one MCP server over any
:class:`~kinetic_sdk.mcp.transport.Transport` (stdio, SSE, ...). It does not
know or care which transport it was given — construction and lifecycle of
the transport belong to the caller (usually
:class:`~kinetic_sdk.mcp.registry.MCPServerRegistry`).

Connection lifecycle — the order is mandated by the MCP spec and enforced
here:

1. Client sends an ``initialize`` request (``protocolVersion``,
   ``capabilities``, ``clientInfo``).
2. Server answers with its own ``protocolVersion``/``capabilities``/
   ``serverInfo``.
3. Client sends the ``notifications/initialized`` notification.
4. ONLY after step 3 may ``tools/list`` / ``tools/call`` be used — calling
   them earlier raises :class:`MCPClientError` (an internal ``_initialized``
   flag tracks the handshake).

Every request carries a fresh id from
:class:`~kinetic_sdk.mcp.protocol.RequestIdGenerator` and the response is
matched against that id — a mismatched id is a protocol violation and
raises, it is never silently paired up. Interleaved notifications are
skipped (optionally surfaced via the ``on_notification`` callback);
server-initiated requests (sampling, roots, ...) are not supported by this
version and are ignored.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from kinetic_sdk.mcp.protocol import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    RequestIdGenerator,
)
from kinetic_sdk.mcp.transport import MCPTimeoutError, Transport

#: Protocol revision this client speaks (2025-03-26 is the widely deployed
#: one; servers negotiate down/up via the initialize exchange).
MCP_PROTOCOL_VERSION = "2025-03-26"


class MCPClientError(RuntimeError):
    """Client-side MCP failure: bad handshake, call before initialize, ..."""


class MCPHandshakeError(MCPClientError):
    """The server answered ``initialize`` with a non-compliant response."""


class MCPServerError(MCPClientError):
    """The server returned a JSON-RPC error for one of our requests."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP server error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPClient:
    """Synchronous MCP client over an arbitrary transport.

    Args:
        transport: A connected (or lazily connecting) transport. Ownership
            transfers to the client: :meth:`close` closes it.
        init_timeout: Overall deadline (seconds) for the ``initialize``
            handshake. Parameterised so tests and slow servers can tune it.
        request_timeout: Default per-request deadline for
            ``tools/list`` / ``tools/call``.
        client_name / client_version: Reported to the server in
            ``clientInfo`` during the handshake.
        on_notification: Optional callback invoked with every server
            notification received while waiting for a response (progress,
            log messages, ...).
    """

    def __init__(
        self,
        transport: Transport,
        init_timeout: float = 10.0,
        request_timeout: float = 30.0,
        client_name: str = "kinetic-agent-sdk",
        client_version: str = "0.1.0",
        on_notification: Callable[[JsonRpcNotification], None] | None = None,
    ) -> None:
        self._transport = transport
        self._init_timeout = init_timeout
        self._request_timeout = request_timeout
        self._client_info = {"name": client_name, "version": client_version}
        self._on_notification = on_notification
        self._ids = RequestIdGenerator()
        self._initialized = False
        #: Populated by a successful handshake: the server's protocolVersion.
        self.protocol_version: str | None = None
        #: Populated by a successful handshake: serverInfo + capabilities.
        self.server_info: dict[str, Any] | None = None

    @property
    def initialized(self) -> bool:
        """True once the 3-step handshake completed."""
        return self._initialized

    # -- handshake -------------------------------------------------------------

    def initialize(self, timeout: float | None = None) -> dict[str, Any]:
        """Run the mandatory 3-step handshake. Returns the server's result.

        Idempotent: a second call returns the cached handshake result without
        touching the wire.

        Raises:
            MCPTimeoutError: The server did not answer in time.
            MCPHandshakeError: The answer misses a compatible
                ``protocolVersion`` or is otherwise malformed.
            MCPServerError: The server explicitly rejected the handshake.
        """
        if self._initialized:
            assert self.server_info is not None
            return self.server_info
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": dict(self._client_info),
            },
            timeout if timeout is not None else self._init_timeout,
        )
        if not isinstance(result, dict) or not isinstance(result.get("protocolVersion"), str):
            raise MCPHandshakeError(
                "Server's initialize response misses a string 'protocolVersion': "
                f"{result!r}"
            )
        self.protocol_version = result["protocolVersion"]
        self.server_info = result
        self._transport.send(JsonRpcNotification(method="notifications/initialized"))
        self._initialized = True
        return result

    # -- operations --------------------------------------------------------------

    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Return the server's tool schemas (MCP shape: ``inputSchema``).

        Raises:
            MCPClientError: Called before :meth:`initialize` completed.
        """
        self._require_initialized("list_tools")
        result = self._request("tools/list", {}, timeout or self._request_timeout)
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise MCPClientError(
                f"Malformed tools/list response: expected a 'tools' list, got {result!r}"
            )
        return result["tools"]

    def call_tool(
        self, name: str, arguments: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """Invoke one tool on the server and return the raw MCP result dict.

        The result follows the MCP shape ``{"content": [...], "isError":
        bool}``; mapping it onto :class:`~kinetic_sdk.tool.base.ToolResult`
        is :class:`~kinetic_sdk.mcp.adapter.MCPToolAdapter`'s job.

        Raises:
            MCPClientError: Called before :meth:`initialize` completed.
            MCPServerError: The server rejected the call at protocol level
                (unknown tool, invalid params, ...).
        """
        self._require_initialized("call_tool")
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout or self._request_timeout,
        )

    def close(self) -> None:
        """Close the underlying transport. Idempotent."""
        self._transport.close()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals -----------------------------------------------------------------

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise MCPClientError(
                f"Cannot call {operation}() before initialize() completed — "
                "the MCP handshake (initialize + notifications/initialized) "
                "must run first."
            )

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        request_id = self._ids.next()
        self._transport.send(JsonRpcRequest(id=request_id, method=method, params=params))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPTimeoutError(
                    f"MCP server did not answer {method!r} (id={request_id}) "
                    f"within {timeout}s"
                )
            msg = self._transport.receive(timeout=remaining)
            if isinstance(msg, JsonRpcNotification):
                if self._on_notification is not None:
                    self._on_notification(msg)
                continue
            if isinstance(msg, JsonRpcRequest):
                # Server-initiated requests (sampling, roots, ...) are not
                # supported in this version; ignore and keep waiting.
                continue
            assert isinstance(msg, JsonRpcResponse)
            if msg.id != request_id:
                raise MCPClientError(
                    f"Response id mismatch for {method!r}: got id={msg.id!r}, "
                    f"expected id={request_id!r} — the server is not pairing "
                    "responses to requests correctly."
                )
            if msg.is_error:
                assert msg.error is not None
                raise MCPServerError(
                    code=msg.error["code"],
                    message=msg.error["message"],
                    data=msg.error.get("data"),
                )
            return msg.result
