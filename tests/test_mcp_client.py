"""Tests for kinetic_sdk.mcp.client (MCPClient handshake + requests)."""

from __future__ import annotations

import queue
import sys
from typing import Callable

import pytest

from kinetic_sdk.mcp.client import (
    MCPClient,
    MCPClientError,
    MCPHandshakeError,
    MCPServerError,
)
from kinetic_sdk.mcp.protocol import (
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)
from kinetic_sdk.mcp.transport import MCPTimeoutError, StdioTransport, Transport

from ._mcp_fakes import FAKE_MCP_SERVER_SCRIPT


class FakeTransport(Transport):
    """In-memory transport driven by a responder callback.

    ``responder(message)`` returns a list of messages to feed back through
    ``receive`` (in order). No responder -> ``receive`` blocks until timeout.
    """

    def __init__(
        self,
        responder: Callable[[JsonRpcMessage], list[JsonRpcMessage]] | None = None,
    ) -> None:
        self.sent: list[JsonRpcMessage] = []
        self.closed = False
        self._responder = responder
        self._inbox: queue.Queue[JsonRpcMessage] = queue.Queue()

    def send(self, message: JsonRpcMessage) -> None:
        self.sent.append(message)
        if self._responder is not None:
            for reply in self._responder(message):
                self._inbox.put(reply)

    def receive(self, timeout: float | None = None) -> JsonRpcMessage:
        try:
            return self._inbox.get(timeout=timeout if timeout is not None else 60)
        except queue.Empty:
            raise MCPTimeoutError("fake transport inbox empty") from None

    def close(self) -> None:
        self.closed = True


def ok_initialize_responder(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
    if isinstance(msg, JsonRpcRequest) and msg.method == "initialize":
        return [
            JsonRpcResponse(
                id=msg.id,
                result={
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "unit-fake", "version": "1.0"},
                },
            )
        ]
    if isinstance(msg, JsonRpcRequest) and msg.method == "tools/list":
        return [
            JsonRpcResponse(
                id=msg.id,
                result={"tools": [{"name": "search", "description": "d", "inputSchema": {}}]},
            )
        ]
    if isinstance(msg, JsonRpcRequest) and msg.method == "tools/call":
        return [
            JsonRpcResponse(
                id=msg.id,
                result={"content": [{"type": "text", "text": "done"}], "isError": False},
            )
        ]
    return []


class TestHandshake:
    def test_handshake_order_is_initialize_then_notification_then_calls(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        client = MCPClient(transport)
        client.initialize()
        client.list_tools()
        methods = [
            m.method for m in transport.sent if isinstance(m, (JsonRpcRequest, JsonRpcNotification))
        ]
        assert methods == ["initialize", "notifications/initialized", "tools/list"]

    def test_initialize_stores_server_info_and_version(self) -> None:
        client = MCPClient(FakeTransport(ok_initialize_responder))
        result = client.initialize()
        assert client.initialized
        assert client.protocol_version == "2025-03-26"
        assert result["serverInfo"] == {"name": "unit-fake", "version": "1.0"}
        assert client.server_info is result

    def test_initialize_is_idempotent(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        client = MCPClient(transport)
        first = client.initialize()
        second = client.initialize()
        assert first is second
        assert len([m for m in transport.sent if isinstance(m, JsonRpcRequest)]) == 1

    def test_initialize_sends_protocol_version_and_client_info(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        client = MCPClient(transport, client_name="kinetic-test", client_version="9.9")
        client.initialize()
        init_req = transport.sent[0]
        assert isinstance(init_req, JsonRpcRequest)
        assert init_req.params["protocolVersion"] == "2025-03-26"
        assert init_req.params["clientInfo"] == {"name": "kinetic-test", "version": "9.9"}

    def test_initialize_timeout(self) -> None:
        client = MCPClient(FakeTransport(responder=None), init_timeout=0.2)
        with pytest.raises(MCPTimeoutError):
            client.initialize()
        assert not client.initialized

    def test_initialize_rejects_response_without_protocol_version(self) -> None:
        def bad(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest):
                return [JsonRpcResponse(id=msg.id, result={"capabilities": {}})]
            return []

        client = MCPClient(FakeTransport(bad))
        with pytest.raises(MCPHandshakeError, match="protocolVersion"):
            client.initialize()
        assert not client.initialized

    def test_initialize_server_error_propagates(self) -> None:
        def refusing(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest):
                return [
                    JsonRpcResponse.make_error(msg.id, code=-32000, message="go away")
                ]
            return []

        client = MCPClient(FakeTransport(refusing))
        with pytest.raises(MCPServerError) as exc_info:
            client.initialize()
        assert exc_info.value.code == -32000
        assert "go away" in str(exc_info.value)


class TestCallsRequireInitialize:
    def test_list_tools_before_initialize_raises(self) -> None:
        client = MCPClient(FakeTransport(ok_initialize_responder))
        with pytest.raises(MCPClientError, match="before initialize"):
            client.list_tools()

    def test_call_tool_before_initialize_raises(self) -> None:
        client = MCPClient(FakeTransport(ok_initialize_responder))
        with pytest.raises(MCPClientError, match="before initialize"):
            client.call_tool("search", {})


class TestRequestResponseCorrelation:
    def test_mismatched_response_id_raises(self) -> None:
        def confused(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest):
                return [JsonRpcResponse(id=msg.id + 1000, result={})]
            return []

        client = MCPClient(FakeTransport(confused))
        with pytest.raises(MCPClientError, match="id mismatch"):
            client.initialize()

    def test_interleaved_notifications_are_skipped(self) -> None:
        seen: list[str] = []

        def chatty(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest):
                return [
                    JsonRpcNotification(method="notifications/progress", params={"pct": 50}),
                    JsonRpcResponse(
                        id=msg.id,
                        result={
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "serverInfo": {"name": "x", "version": "1"},
                        },
                    ),
                ]
            return []

        client = MCPClient(
            FakeTransport(chatty), on_notification=lambda n: seen.append(n.method)
        )
        client.initialize()
        assert seen == ["notifications/progress"]

    def test_server_initiated_requests_are_ignored(self) -> None:
        def sampling(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest):
                return [
                    JsonRpcRequest(id="srv-1", method="sampling/createMessage", params={}),
                    JsonRpcResponse(
                        id=msg.id,
                        result={
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "serverInfo": {"name": "x", "version": "1"},
                        },
                    ),
                ]
            return []

        client = MCPClient(FakeTransport(sampling))
        client.initialize()
        assert client.initialized

    def test_request_ids_increase_across_calls(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        client = MCPClient(transport)
        client.initialize()
        client.list_tools()
        client.call_tool("search", {"q": "x"})
        request_ids = [m.id for m in transport.sent if isinstance(m, JsonRpcRequest)]
        assert request_ids == sorted(request_ids)
        assert len(set(request_ids)) == len(request_ids)


class TestOperations:
    def test_list_tools_returns_tool_list(self) -> None:
        client = MCPClient(FakeTransport(ok_initialize_responder))
        client.initialize()
        tools = client.list_tools()
        assert tools == [{"name": "search", "description": "d", "inputSchema": {}}]

    def test_list_tools_malformed_response_raises(self) -> None:
        def weird(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest) and msg.method == "initialize":
                return ok_initialize_responder(msg)
            if isinstance(msg, JsonRpcRequest):
                return [JsonRpcResponse(id=msg.id, result={"noTools": []})]
            return []

        client = MCPClient(FakeTransport(weird))
        client.initialize()
        with pytest.raises(MCPClientError, match="tools"):
            client.list_tools()

    def test_call_tool_passes_name_and_arguments(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        client = MCPClient(transport)
        client.initialize()
        result = client.call_tool("search", {"q": "abc"})
        call = transport.sent[-1]
        assert isinstance(call, JsonRpcRequest)
        assert call.method == "tools/call"
        assert call.params == {"name": "search", "arguments": {"q": "abc"}}
        assert result["content"][0]["text"] == "done"

    def test_call_tool_server_error_raises_with_details(self) -> None:
        def failing(msg: JsonRpcMessage) -> list[JsonRpcMessage]:
            if isinstance(msg, JsonRpcRequest) and msg.method == "initialize":
                return ok_initialize_responder(msg)
            if isinstance(msg, JsonRpcRequest):
                return [
                    JsonRpcResponse.make_error(
                        msg.id, code=-32602, message="unknown tool", data={"tool": "nope"}
                    )
                ]
            return []

        client = MCPClient(FakeTransport(failing))
        client.initialize()
        with pytest.raises(MCPServerError) as exc_info:
            client.call_tool("nope", {})
        assert exc_info.value.code == -32602
        assert exc_info.value.data == {"tool": "nope"}

    def test_close_closes_transport_and_context_manager(self) -> None:
        transport = FakeTransport(ok_initialize_responder)
        with MCPClient(transport) as client:
            client.initialize()
        assert transport.closed


class TestRealStdioIntegration:
    """The same client, but over a real subprocess via StdioTransport."""

    def test_full_flow_against_fake_server_subprocess(self, tmp_path) -> None:
        script = tmp_path / "fake_server.py"
        script.write_text(FAKE_MCP_SERVER_SCRIPT)
        transport = StdioTransport(command=sys.executable, args=[str(script)])
        with MCPClient(transport, init_timeout=10) as client:
            info = client.initialize()
            assert info["serverInfo"]["name"] == "fake-mcp"
            tools = client.list_tools()
            assert {t["name"] for t in tools} == {"echo", "fail", "crash"}
            result = client.call_tool("echo", {"text": "xin chào"})
            assert result["isError"] is False
            assert result["content"][0]["text"] == "xin chào"
            error_result = client.call_tool("fail", {})
            assert error_result["isError"] is True
