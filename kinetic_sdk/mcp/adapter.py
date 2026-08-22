"""Adapt MCP server tools into Kinetic :class:`Tool` objects (Stage 4).

:class:`MCPToolAdapter` wraps ONE tool exposed by ONE MCP server behind the
standard :class:`~kinetic_sdk.tool.base.Tool` interface, so the agent loop
can call it exactly like a built-in tool — permission policy, audit log and
hooks all apply unchanged.

Two safety properties live here:

* **Name prefixing** — the registered tool name is always
  ``"{server_name}.{tool_name}"`` (e.g. ``"unity.search"``). Two servers
  both exposing a ``search`` tool therefore never clobber each other when
  their tools are merged into one agent's tool list.
* **Output redaction** — everything coming back from an external MCP server
  (whose provenance the SDK cannot vouch for) is scrubbed through
  :func:`~kinetic_sdk.security.redact.redact_value` before it reaches the
  conversation, the event bus or the audit log.

MCP-level failures (transport dead, server error, ``isError: true``) are
mapped to ``ToolResult(error=...)`` — the adapter never raises into the
agent loop for an expected failure mode.
"""

from __future__ import annotations

import json
from typing import Any

from kinetic_sdk.mcp.client import MCPClient, MCPClientError
from kinetic_sdk.mcp.transport import MCPTransportError
from kinetic_sdk.security.redact import redact_value
from kinetic_sdk.tool.base import Tool, ToolResult


class MCPToolAdapter(Tool):
    """One MCP server's tool, exposed as a Kinetic ``Tool``.

    Args:
        mcp_client: An INITIALIZED :class:`MCPClient` connected to the
            server that owns the tool.
        tool_name: The tool's name on the server (e.g. ``"search"``).
        server_name: Short name of the server (e.g. ``"unity"``); used as
            the prefix of the registered tool name.
        description: Human-readable description (from ``tools/list``).
        input_schema: JSON schema for the arguments (from ``tools/list``'s
            ``inputSchema`` field). ``None`` degrades to an empty object
            schema.
        timeout: Per-call deadline forwarded to
            :meth:`MCPClient.call_tool`; ``None`` uses the client's default.

    The registered :attr:`name` is ``"{server_name}.{tool_name}"`` — see the
    module docstring for why the prefix is mandatory.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        tool_name: str,
        server_name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        if not server_name:
            raise ValueError("server_name must be non-empty")
        self._client = mcp_client
        self._tool_name = tool_name
        self._server_name = server_name
        self._timeout = timeout
        # Instance attributes shadow the ClassVar annotations on Tool — the
        # same pattern MockTool uses for per-instance schemas.
        self.name = f"{server_name}.{tool_name}"
        self.description = description or f"MCP tool {tool_name!r} from server {server_name!r}"
        self.parameters = input_schema or {"type": "object", "properties": {}}

    @classmethod
    def from_mcp_schema(
        cls,
        mcp_client: MCPClient,
        server_name: str,
        schema: dict[str, Any],
        timeout: float | None = None,
    ) -> "MCPToolAdapter":
        """Build an adapter from one entry of a ``tools/list`` response.

        MCP names the schema field ``inputSchema`` (camelCase); Kinetic's
        ``Tool`` interface calls it ``parameters`` / ``input_schema``.
        """
        if not isinstance(schema.get("name"), str) or not schema["name"]:
            raise ValueError(f"MCP tool schema misses a string 'name': {schema!r}")
        return cls(
            mcp_client=mcp_client,
            tool_name=schema["name"],
            server_name=server_name,
            description=schema.get("description") or "",
            input_schema=schema.get("inputSchema"),
            timeout=timeout,
        )

    @property
    def mcp_tool_name(self) -> str:
        """The unprefixed tool name on the remote server."""
        return self._tool_name

    @property
    def server_name(self) -> str:
        return self._server_name

    def execute(self, **params: Any) -> ToolResult:
        """Call the remote tool, mapping every failure mode to ToolResult."""
        metadata = {"mcp_server": self._server_name, "mcp_tool": self._tool_name}
        try:
            result = self._client.call_tool(self._tool_name, params, timeout=self._timeout)
        except (MCPClientError, MCPTransportError) as exc:
            return ToolResult(
                error=f"MCP tool {self.name!r} call failed: {exc}",
                metadata=metadata,
            )
        if not isinstance(result, dict):
            return ToolResult(
                error=f"MCP tool {self.name!r} returned a malformed result: {result!r}",
                metadata=metadata,
            )

        output = self._extract_output(result.get("content"))
        output = redact_value(output)  # external server output is untrusted
        if result.get("isError"):
            return ToolResult(
                error=self._as_text(output) or "MCP server reported a tool error",
                metadata=metadata,
            )
        return ToolResult(output=output, metadata=metadata)

    @staticmethod
    def _extract_output(content: Any) -> Any:
        """Flatten MCP ``content`` blocks into a compact ToolResult output.

        A single text block becomes a plain string; anything else (multiple
        blocks, images, resources) stays a list of the raw blocks so no
        information is lost.
        """
        if not isinstance(content, list) or not content:
            return ""
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            return content[0].get("text", "")
        return content

    @staticmethod
    def _as_text(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(output)
