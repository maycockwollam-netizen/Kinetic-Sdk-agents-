"""Tests for kinetic_sdk.mcp.server (MCPServer — Kinetic as MCP server)."""

from __future__ import annotations

import os
import threading

import pytest

from kinetic_sdk.mcp.client import MCPClient
from kinetic_sdk.mcp.protocol import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)
from kinetic_sdk.mcp.server import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    SERVER_NOT_INITIALIZED,
    MCPServer,
    PromptArgument,
    ServerPrompt,
)
from kinetic_sdk.mcp.transport import StdioTransport
from kinetic_sdk.security import AllowListPolicy, InMemoryAuditLogger, PermissivePolicy
from kinetic_sdk.testing import MockTool
from kinetic_sdk.tool.base import ToolResult


def _make_server(**kwargs) -> MCPServer:
    tools = [
        MockTool(name="echo", description="Echo text", parameters={"type": "object"},
                 handler=lambda text="": ToolResult(output=f"echo:{text}")),
        MockTool(name="danger", description="Dangerous", parameters={"type": "object"},
                 result=ToolResult(output="done")),
    ]
    return MCPServer(tools, **kwargs)


def _handshake(server: MCPServer) -> None:
    server.handle_message(
        JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": "2025-03-26"})
    )
    server.handle_message(JsonRpcNotification(method="notifications/initialized"))


class TestHandshake:
    def test_initialize_returns_server_info(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        response = server.handle_message(
            JsonRpcRequest(
                id=1,
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "claude-desktop", "version": "1"},
                },
            )
        )
        assert isinstance(response, JsonRpcResponse)
        assert response.id == 1
        assert response.result["protocolVersion"] == "2025-03-26"
        assert response.result["serverInfo"]["name"] == "kinetic-agent-sdk"
        assert "tools" in response.result["capabilities"]
        assert server.client_protocol_version == "2024-11-05"
        assert server.client_info == {"name": "claude-desktop", "version": "1"}

    def test_initialize_without_protocol_version_rejected(self) -> None:
        server = _make_server()
        response = server.handle_message(JsonRpcRequest(id=1, method="initialize", params={}))
        assert response.is_error
        assert response.error["code"] == INVALID_PARAMS

    def test_tools_list_refused_before_initialized_notification(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        server.handle_message(
            JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": "2025-03-26"})
        )
        response = server.handle_message(JsonRpcRequest(id=2, method="tools/list"))
        assert response.is_error
        assert response.error["code"] == SERVER_NOT_INITIALIZED

    def test_tools_call_refused_before_any_handshake(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        response = server.handle_message(
            JsonRpcRequest(id=1, method="tools/call", params={"name": "echo", "arguments": {}})
        )
        assert response.is_error
        assert response.error["code"] == SERVER_NOT_INITIALIZED

    def test_notification_produces_no_response(self) -> None:
        server = _make_server()
        assert (
            server.handle_message(JsonRpcNotification(method="notifications/initialized"))
            is None
        )

    def test_ping_anytime(self) -> None:
        server = _make_server()
        response = server.handle_message(JsonRpcRequest(id=1, method="ping"))
        assert response.result == {}

    def test_unknown_method_error(self) -> None:
        server = _make_server()
        _handshake(server)
        response = server.handle_message(JsonRpcRequest(id=9, method="completions/get"))
        assert response.is_error
        assert response.error["code"] == METHOD_NOT_FOUND


class TestToolsList:
    def test_returns_mcp_shaped_schemas(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        _handshake(server)
        response = server.handle_message(JsonRpcRequest(id=2, method="tools/list"))
        tools = response.result["tools"]
        assert {t["name"] for t in tools} == {"echo", "danger"}
        echo = next(t for t in tools if t["name"] == "echo")
        assert echo["description"] == "Echo text"
        assert "inputSchema" in echo  # MCP camelCase, not Kinetic's input_schema

    def test_duplicate_tool_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            MCPServer(
                [MockTool(name="x", result=1), MockTool(name="x", result=2)],
                permission_policy=PermissivePolicy(),
            )


class TestToolsCall:
    def test_allowed_call_executes_and_returns_content(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(
                id=3, method="tools/call",
                params={"name": "echo", "arguments": {"text": "hi"}},
            )
        )
        assert not response.is_error
        assert response.result["isError"] is False
        assert response.result["content"] == [{"type": "text", "text": "echo:hi"}]

    def test_unknown_tool_is_invalid_params(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "ghost", "arguments": {}})
        )
        assert response.is_error
        assert response.error["code"] == INVALID_PARAMS

    def test_missing_arguments_rejected(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "echo"})
        )
        assert response.is_error
        assert response.error["code"] == INVALID_PARAMS

    def test_tool_exception_becomes_is_error_result(self) -> None:
        def boom() -> ToolResult:
            raise RuntimeError("kaboom")

        server = MCPServer(
            [MockTool(name="fragile", handler=boom)],
            permission_policy=PermissivePolicy(),
        )
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "fragile", "arguments": {}})
        )
        assert response.result["isError"] is True
        assert "kaboom" in response.result["content"][0]["text"]

    def test_tool_error_result_propagates_as_is_error(self) -> None:
        server = MCPServer(
            [MockTool(name="sad", result=ToolResult(error="went wrong"))],
            permission_policy=PermissivePolicy(),
        )
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "sad", "arguments": {}})
        )
        assert response.result["isError"] is True
        assert "went wrong" in response.result["content"][0]["text"]


class TestSecurity:
    def test_default_policy_denies_everything(self) -> None:
        server = _make_server()  # default: empty AllowListPolicy
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(
                id=3, method="tools/call",
                params={"name": "echo", "arguments": {"text": "hi"}},
            )
        )
        assert response.result["isError"] is True
        assert "Permission denied" in response.result["content"][0]["text"]

    def test_denied_tool_is_not_executed(self) -> None:
        tool = MockTool(name="echo", result=ToolResult(output="ran"))
        server = MCPServer([tool], permission_policy=AllowListPolicy(always_allow=[]))
        _handshake(server)
        server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "echo", "arguments": {}})
        )
        assert tool.calls == []  # never reached execute()

    def test_allowlist_allows_only_listed_tools(self) -> None:
        server = _make_server(
            permission_policy=AllowListPolicy(always_allow=["echo"]),
        )
        _handshake(server)
        ok = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "echo", "arguments": {}})
        )
        denied = server.handle_message(
            JsonRpcRequest(id=4, method="tools/call", params={"name": "danger", "arguments": {}})
        )
        assert ok.result["isError"] is False
        assert denied.result["isError"] is True
        assert "Permission denied" in denied.result["content"][0]["text"]

    def test_requires_confirmation_is_denied_like_agent_loop(self) -> None:
        server = _make_server(
            permission_policy=AllowListPolicy(
                always_allow=["danger"],
                require_confirmation_patterns={"danger": [".*"]},
            ),
        )
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "danger", "arguments": {}})
        )
        assert response.result["isError"] is True
        assert "requires manual confirmation" in response.result["content"][0]["text"]

    def test_audit_log_records_calls_and_denials(self) -> None:
        audit = InMemoryAuditLogger()
        server = _make_server(
            permission_policy=AllowListPolicy(always_allow=["echo"]),
            audit_logger=audit,
        )
        _handshake(server)
        server.handle_message(
            JsonRpcRequest(
                id=3, method="tools/call",
                params={"name": "echo", "arguments": {"text": "hi"}},
            )
        )
        server.handle_message(
            JsonRpcRequest(id=4, method="tools/call", params={"name": "danger", "arguments": {}})
        )
        events = [e["event"] for e in audit.entries]
        assert events == ["tool_call", "tool_result", "tool_call", "permission_denied"]

    def test_secret_in_tool_output_is_redacted_before_serving(self) -> None:
        token = "ghp_" + "a1b2c3d4e5f6" * 3
        server = MCPServer(
            [MockTool(name="leak", result=ToolResult(output=f"token={token}"))],
            permission_policy=PermissivePolicy(),
        )
        _handshake(server)
        response = server.handle_message(
            JsonRpcRequest(id=3, method="tools/call", params={"name": "leak", "arguments": {}})
        )
        text = response.result["content"][0]["text"]
        assert token not in text
        assert "[REDACTED]" in text


class TestServeOverPipes:
    """serve_forever driven by a REAL MCPClient over real OS pipes."""

    def test_full_round_trip_client_to_kinetic_server(self) -> None:
        to_server_r, to_server_w = os.pipe()
        from_server_r, from_server_w = os.pipe()
        server = _make_server(permission_policy=PermissivePolicy())
        server_transport = StdioTransport.attach(
            reader=os.fdopen(to_server_r, "rb"), writer=os.fdopen(from_server_w, "wb")
        )
        thread = threading.Thread(
            target=server.serve_forever, args=(server_transport,), daemon=True
        )
        thread.start()

        client_writer = os.fdopen(to_server_w, "wb")
        client_transport = StdioTransport.attach(
            reader=os.fdopen(from_server_r, "rb"), writer=client_writer
        )
        client = MCPClient(client_transport, init_timeout=5, request_timeout=5)
        try:
            info = client.initialize()
            assert info["serverInfo"]["name"] == "kinetic-agent-sdk"
            tools = client.list_tools()
            assert {t["name"] for t in tools} == {"echo", "danger"}
            result = client.call_tool("echo", {"text": "từ client bên ngoài"})
            assert result["isError"] is False
            assert result["content"][0]["text"] == "echo:từ client bên ngoài"
        finally:
            client.close()
            # attach() deliberately does not close host-owned streams, so the
            # test closes the writer itself to signal EOF to the server.
            client_writer.close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_malformed_line_gets_parse_error_and_server_survives(self) -> None:
        to_server_r, to_server_w = os.pipe()
        from_server_r, from_server_w = os.pipe()
        server = _make_server(permission_policy=PermissivePolicy())
        server_transport = StdioTransport.attach(
            reader=os.fdopen(to_server_r, "rb"), writer=os.fdopen(from_server_w, "wb")
        )
        thread = threading.Thread(
            target=server.serve_forever, args=(server_transport,), daemon=True
        )
        thread.start()
        client_writer = os.fdopen(to_server_w, "wb")
        client_transport = StdioTransport.attach(
            reader=os.fdopen(from_server_r, "rb"), writer=client_writer
        )
        try:
            # Raw garbage on the wire -> -32700, then the server keeps serving.
            client_transport._writer.write(b"this is not json\n")
            client_transport._writer.flush()
            msg = client_transport.receive(timeout=5)
            assert isinstance(msg, JsonRpcResponse)
            assert msg.error["code"] == -32700

            client = MCPClient(client_transport, init_timeout=5)
            info = client.initialize()
            assert info["serverInfo"]["name"] == "kinetic-agent-sdk"
        finally:
            client_transport.close()
            client_writer.close()  # EOF is the server's shutdown signal
        thread.join(timeout=5)
        assert not thread.is_alive()


class TestResourcesPrompts:
    def _server(self) -> MCPServer:
        return _make_server(
            permission_policy=PermissivePolicy(),
            resources={"docs://readme": "# Readme\nHello world"},
            prompts={
                "review": ServerPrompt(
                    description="Review code",
                    arguments=(PromptArgument(name="language", required=True),),
                    template="Review this {language} code",
                ),
                "nowhite": ServerPrompt(description="No args", template="plain {extra}"),
            },
        )

    def test_initialize_declares_capabilities_when_present(self) -> None:
        response = self._server().handle_message(
            JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": "2025-03-26"})
        )
        caps = response.result["capabilities"]  # type: ignore[union-attr]
        assert "resources" in caps and "prompts" in caps

    def test_initialize_omits_capabilities_when_absent(self) -> None:
        server = _make_server(permission_policy=PermissivePolicy())
        response = server.handle_message(
            JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": "2025-03-26"})
        )
        caps = response.result["capabilities"]  # type: ignore[union-attr]
        assert "resources" not in caps and "prompts" not in caps

    def test_resources_list_and_read(self) -> None:
        server = self._server()
        _handshake(server)
        listed = server.handle_message(JsonRpcRequest(id=1, method="resources/list"))
        assert listed.result["resources"] == [  # type: ignore[union-attr]
            {"uri": "docs://readme", "name": "readme", "mimeType": "text/plain"}
        ]
        read = server.handle_message(
            JsonRpcRequest(id=2, method="resources/read", params={"uri": "docs://readme"})
        )
        assert read.result["contents"][0]["text"] == "# Readme\nHello world"  # type: ignore[union-attr]
        miss = server.handle_message(
            JsonRpcRequest(id=3, method="resources/read", params={"uri": "docs://nope"})
        )
        assert miss.is_error and miss.error["code"] == INVALID_PARAMS  # type: ignore[index]

    def test_prompts_list_and_get(self) -> None:
        server = self._server()
        _handshake(server)
        listed = server.handle_message(JsonRpcRequest(id=1, method="prompts/list"))
        prompts = listed.result["prompts"]  # type: ignore[union-attr]
        assert {p["name"] for p in prompts} == {"review", "nowhite"}
        assert next(p for p in prompts if p["name"] == "review")["arguments"][0]["required"] is True

        got = server.handle_message(
            JsonRpcRequest(id=2, method="prompts/get", params={"name": "review", "arguments": {"language": "python"}})
        )
        assert got.result["messages"][0]["content"]["text"] == "Review this python code"  # type: ignore[union-attr]

        missing = server.handle_message(JsonRpcRequest(id=3, method="prompts/get", params={"name": "review"}))
        assert missing.is_error and "language" in missing.error["message"]  # type: ignore[index]

        unknown = server.handle_message(JsonRpcRequest(id=4, method="prompts/get", params={"name": "nope"}))
        assert unknown.is_error

    def test_resources_and_prompts_gated_to_ready(self) -> None:
        server = self._server()  # no handshake
        for method in ("resources/list", "resources/read", "prompts/list", "prompts/get"):
            response = server.handle_message(JsonRpcRequest(id=9, method=method, params={}))
            assert response.is_error and response.error["code"] == SERVER_NOT_INITIALIZED  # type: ignore[index]
