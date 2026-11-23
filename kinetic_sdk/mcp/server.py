"""MCP server: expose Kinetic tools to external MCP clients (Stage 4).

:class:`MCPServer` is the mirror image of :class:`MCPClient`: it lets ANOTHER
MCP client (Claude Desktop, an IDE, or a second Kinetic agent) connect to a
running Kinetic instance and call its tools. The typical deployment is a
stdio subprocess — the client spawns Kinetic and speaks newline-delimited
JSON-RPC on its stdin/stdout::

    # entry point of the spawned process
    from kinetic_sdk.mcp.server import MCPServer
    from kinetic_sdk.security import AllowListPolicy

    MCPServer.serve_stdio(
        tools=[GitTool()],
        permission_policy=AllowListPolicy(always_allow=["git"]),
    )

Security is NOT optional on this path: every ``tools/call`` request from the
external client flows through the same ``permission_policy.check(...)`` +
audit logging the internal agent loop applies. A request arriving over MCP
gets no shortcut around :mod:`kinetic_sdk.security` just because it did not
come from the local agent loop. ``requires_confirmation`` decisions are
denied (the server side has no confirmation UX), matching the agent loop's
safe fallback.

The server speaks the mandated handshake: it answers ``initialize``, waits
for ``notifications/initialized``, and refuses ``tools/list`` / ``tools/call``
with error ``-32002`` until that notification arrives.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kinetic_sdk.mcp.client import MCP_PROTOCOL_VERSION
from kinetic_sdk.mcp.protocol import (
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    MCPProtocolError,
)
from kinetic_sdk.mcp.transport import MCPTransportError, StdioTransport, Transport
from kinetic_sdk.security.audit import AuditLogger, InMemoryAuditLogger
from kinetic_sdk.security.policy import AllowListPolicy, PermissionPolicy
from kinetic_sdk.security.redact import redact_value
from kinetic_sdk.tool.base import Tool, ToolResult

logger = logging.getLogger(__name__)

#: JSON-RPC error codes used by the server (JSON-RPC reserved + MCP's).
PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SERVER_NOT_INITIALIZED = -32002

#: Server states, driven by the handshake.
_AWAITING_INITIALIZE = "awaiting_initialize"
_AWAITING_INITIALIZED_NOTIFICATION = "awaiting_initialized_notification"
_READY = "ready"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PromptArgument:
    """One input field a server prompt may require."""

    name: str
    description: str = ""
    required: bool = False


@dataclass(frozen=True)
class ServerPrompt:
    """A server-side prompt: metadata, declared arguments, and a template.

    ``template`` is rendered for ``prompts/get`` by filling ``{name}``
    placeholders with the caller-provided argument values (missing names are
    left as ``{placeholder}`` so the model still sees the intent).
    """

    description: str
    arguments: tuple[PromptArgument, ...] = ()
    template: str = "{prompt}"

    def render(self, values: dict[str, str]) -> str:
        text = self.template
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        return text


class MCPServer:
    """Serve Kinetic tools to an external MCP client.

    Args:
        tools: The tools to expose. Names must be unique.
        permission_policy: Gate for every incoming ``tools/call``. Defaults
            to an EMPTY :class:`AllowListPolicy` (deny-by-default) — the same
            safe default the agent loop uses; opt tools in explicitly.
        audit_logger: Sink for call/result/denial entries.
        server_name / server_version: Reported in ``serverInfo`` during the
            handshake.
    """

    def __init__(
        self,
        tools: list[Tool],
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
        server_name: str = "kinetic-agent-sdk",
        server_version: str = "0.1.0",
        resources: dict[str, str] | None = None,
        prompts: dict[str, ServerPrompt] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name exposed to MCP server: {tool.name!r}")
            self._tools[tool.name] = tool
        self.permission_policy = (
            permission_policy if permission_policy is not None else AllowListPolicy()
        )
        self.audit_logger = audit_logger if audit_logger is not None else InMemoryAuditLogger()
        self._server_info = {"name": server_name, "version": server_version}
        # uri -> text content (served read-only over resources/*).
        self._resources: dict[str, str] = dict(resources or {})
        # prompt name -> definition (served over prompts/*).
        self._prompts: dict[str, ServerPrompt] = dict(prompts or {})
        self._state = _AWAITING_INITIALIZE
        #: The protocol version the connected client proposed (for diagnostics).
        self.client_protocol_version: str | None = None
        self.client_info: dict[str, Any] | None = None

    # -- message handling ----------------------------------------------------

    def handle_message(self, message: JsonRpcMessage) -> JsonRpcResponse | None:
        """Handle ONE incoming message; returns the response to send, if any.

        Notifications never produce a response. Requests always produce
        exactly one. This method is transport-free, so tests can drive the
        server directly without any pipes.
        """
        if isinstance(message, JsonRpcNotification):
            self._handle_notification(message)
            return None
        if not isinstance(message, JsonRpcRequest):
            return None  # a response TO us: we never send requests; ignore

        handler = {
            "initialize": self._on_initialize,
            "ping": self._on_ping,
            "tools/list": self._on_tools_list,
            "tools/call": self._on_tools_call,
            "resources/list": self._on_resources_list,
            "resources/read": self._on_resources_read,
            "prompts/list": self._on_prompts_list,
            "prompts/get": self._on_prompts_get,
        }.get(message.method)
        if handler is None:
            return JsonRpcResponse.make_error(
                message.id, METHOD_NOT_FOUND, f"Method not found: {message.method!r}"
            )
        return handler(message)

    def _handle_notification(self, notification: JsonRpcNotification) -> None:
        if (
            notification.method == "notifications/initialized"
            and self._state == _AWAITING_INITIALIZED_NOTIFICATION
        ):
            self._state = _READY

    def _require_ready(self, request: JsonRpcRequest) -> JsonRpcResponse | None:
        """Return an error response unless the handshake fully completed."""
        if self._state != _READY:
            return JsonRpcResponse.make_error(
                request.id,
                SERVER_NOT_INITIALIZED,
                "Server not initialized: complete the initialize + "
                "notifications/initialized handshake first",
            )
        return None

    # -- method handlers -------------------------------------------------------

    def _on_initialize(self, request: JsonRpcRequest) -> JsonRpcResponse:
        params = request.params if isinstance(request.params, dict) else {}
        client_version = params.get("protocolVersion")
        if not isinstance(client_version, str):
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, "initialize params miss 'protocolVersion'"
            )
        self.client_protocol_version = client_version
        client_info = params.get("clientInfo")
        self.client_info = client_info if isinstance(client_info, dict) else None
        self._state = _AWAITING_INITIALIZED_NOTIFICATION
        capabilities: dict[str, Any] = {"tools": {"listChanged": False}}
        if self._resources:
            capabilities["resources"] = {"subscribe": False, "listChanged": False}
        if self._prompts:
            capabilities["prompts"] = {"listChanged": False}
        return JsonRpcResponse(
            id=request.id,
            result={
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": capabilities,
                "serverInfo": dict(self._server_info),
            },
        )

    def _on_ping(self, request: JsonRpcRequest) -> JsonRpcResponse:
        return JsonRpcResponse(id=request.id, result={})

    def _on_resources_list(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        items = [
            {
                "uri": uri,
                "name": uri.rstrip("/").rsplit("/", 1)[-1],
                "mimeType": "text/plain",
            }
            for uri in sorted(self._resources)
        ]
        return JsonRpcResponse(id=request.id, result={"resources": items})

    def _on_resources_read(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        params = request.params if isinstance(request.params, dict) else {}
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, "resources/read params miss a string 'uri'"
            )
        content = self._resources.get(uri)
        if content is None:
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, f"Unknown resource uri: {uri!r}"
            )
        return JsonRpcResponse(
            id=request.id,
            result={
                "contents": [
                    {"uri": uri, "mimeType": "text/plain", "text": content}
                ]
            },
        )

    def _on_prompts_list(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        items = [
            {
                "name": name,
                "description": prompt.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                    }
                    for arg in prompt.arguments
                ],
            }
            for name, prompt in sorted(self._prompts.items())
        ]
        return JsonRpcResponse(id=request.id, result={"prompts": items})

    def _on_prompts_get(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        params = request.params if isinstance(request.params, dict) else {}
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, "prompts/get params miss a string 'name'"
            )
        prompt = self._prompts.get(name)
        if prompt is None:
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, f"Unknown prompt: {name!r}"
            )
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, "'arguments' must be an object"
            )
        missing = [
            arg.name for arg in prompt.arguments if arg.required and arg.name not in arguments
        ]
        if missing:
            return JsonRpcResponse.make_error(
                request.id,
                INVALID_PARAMS,
                f"prompt {name!r} misses required arguments {missing}",
            )
        rendered = prompt.render({str(key): str(value) for key, value in arguments.items()})
        return JsonRpcResponse(
            id=request.id,
            result={
                "description": prompt.description,
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": rendered}}
                ],
            },
        )

    def _on_tools_list(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        return JsonRpcResponse(
            id=request.id,
            result={
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.parameters,
                    }
                    for tool in self._tools.values()
                ]
            },
        )

    def _on_tools_call(self, request: JsonRpcRequest) -> JsonRpcResponse:
        not_ready = self._require_ready(request)
        if not_ready is not None:
            return not_ready
        params = request.params if isinstance(request.params, dict) else {}
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return JsonRpcResponse.make_error(
                request.id,
                INVALID_PARAMS,
                "tools/call params must contain str 'name' and object 'arguments'",
            )
        tool = self._tools.get(name)
        if tool is None:
            return JsonRpcResponse.make_error(
                request.id, INVALID_PARAMS, f"Unknown tool: {name!r}"
            )

        # --- same security path as the internal agent loop, no shortcut ---
        decision = self.permission_policy.check(name, arguments)
        self.audit_logger.log_tool_call(name, arguments, decision, _utcnow())
        if not decision.allowed:
            reason = f"Permission denied: {decision.reason}"
            self.audit_logger.log_permission_denied(name, arguments, reason, _utcnow())
            return self._tool_error_result(request.id, reason)
        if decision.requires_confirmation:
            reason = (
                "requires manual confirmation, not yet supported in automated "
                f"mode ({decision.reason})"
            )
            self.audit_logger.log_permission_denied(name, arguments, reason, _utcnow())
            return self._tool_error_result(request.id, reason)

        try:
            result = tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("MCP-served tool %s raised", name)
            result = ToolResult(error=f"{type(exc).__name__}: {exc}")
        # External clients see redacted output only; the audit log likewise.
        result = ToolResult(
            output=redact_value(result.output),
            error=redact_value(result.error),
            metadata=result.metadata,
        )
        self.audit_logger.log_tool_result(name, result, _utcnow())
        return JsonRpcResponse(
            id=request.id,
            result={
                "content": [{"type": "text", "text": self._result_text(result)}],
                "isError": result.is_error,
            },
        )

    # -- serving -----------------------------------------------------------------

    def serve_forever(self, transport: Transport) -> None:
        """Read/handle/write messages on *transport* until the peer hangs up.

        A dead transport (client closed the pipe) ends the loop cleanly —
        when Kinetic runs as a spawned MCP subprocess, the client closing
        the connection IS the shutdown signal. Malformed JSON lines are
        answered with a ``-32700`` error and the loop continues.
        """
        while True:
            try:
                message = transport.receive()  # block until a message arrives
            except MCPTransportError:
                return  # peer gone: clean shutdown
            except MCPProtocolError as exc:
                try:
                    transport.send(
                        JsonRpcResponse.make_error(None, PARSE_ERROR, str(exc))
                    )
                except MCPTransportError:
                    return
                continue
            response = self.handle_message(message)
            if response is not None:
                try:
                    transport.send(response)
                except MCPTransportError:
                    return

    @classmethod
    def serve_stdio(
        cls,
        tools: list[Tool],
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
        **kwargs: Any,
    ) -> None:
        """Serve over the CURRENT process's stdin/stdout.

        This is the entry point to point an MCP client's ``command`` at,
        e.g. ``python -c "from my_tools import main; main()"`` where main
        calls this. Nothing may print to stdout on this path — stdout is the
        protocol channel.
        """
        server = cls(
            tools,
            permission_policy=permission_policy,
            audit_logger=audit_logger,
            **kwargs,
        )
        server.serve_forever(StdioTransport.attach())

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _tool_error_result(request_id: int | str, message: str) -> JsonRpcResponse:
        """A tool-execution failure, expressed the MCP way (isError result)."""
        return JsonRpcResponse(
            id=request_id,
            result={
                "content": [{"type": "text", "text": message}],
                "isError": True,
            },
        )

    @staticmethod
    def _result_text(result: ToolResult) -> str:
        value = result.error if result.is_error else result.output
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
