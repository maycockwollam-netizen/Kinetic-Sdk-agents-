"""MCP (Model Context Protocol) support — both directions (Stage 4).

**Client** — connect to external MCP servers (Unity, Roblox Studio,
filesystem, GitHub, ...) and use their tools as first-class Kinetic
:class:`~kinetic_sdk.tool.base.Tool` objects inside the agent loop::

    from kinetic_sdk.mcp import MCPServerConfig, MCPServerRegistry
    from kinetic_sdk.secret import SecretRegistry

    secrets = SecretRegistry()  # env-based
    registry = MCPServerRegistry(secrets=secrets)
    registry.register(
        "unity",
        MCPServerConfig.stdio(
            command="unity-mcp",
            args=["--project", "./Game"],
            env={"UNITY_API_KEY": secrets.resolve("UNITY_API_KEY")},
        ),
    )
    tools = registry.get_tools_as_kinetic_tools("unity")  # ["unity.search", ...]
    agent = Agent(llm=..., tools=[*tools, GitTool()], ...)

**Server** — expose Kinetic's own tools to an external MCP client (Claude
Desktop, an IDE, another Kinetic agent) over stdio::

    from kinetic_sdk.mcp import MCPServer
    from kinetic_sdk.security import AllowListPolicy

    MCPServer.serve_stdio(tools=[GitTool()],
                          permission_policy=AllowListPolicy(always_allow=["git"]))

Every incoming ``tools/call`` flows through the same permission policy +
audit log as the internal agent loop — no security shortcut for MCP traffic.
"""

from kinetic_sdk.mcp.adapter import MCPToolAdapter
from kinetic_sdk.mcp.client import (
    MCP_PROTOCOL_VERSION,
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
    MCPProtocolError,
    RequestIdGenerator,
    decode,
    encode,
)
from kinetic_sdk.mcp.registry import MCPServerConfig, MCPServerRegistry
from kinetic_sdk.mcp.server import MCPServer
from kinetic_sdk.mcp.transport import (
    MCPTimeoutError,
    MCPTransportError,
    ProcessFactory,
    SSETransport,
    StdioTransport,
    Transport,
)

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCPClient",
    "MCPClientError",
    "MCPHandshakeError",
    "MCPServerError",
    "MCPServer",
    "MCPServerConfig",
    "MCPServerRegistry",
    "MCPToolAdapter",
    "JsonRpcMessage",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "MCPProtocolError",
    "RequestIdGenerator",
    "decode",
    "encode",
    "MCPTimeoutError",
    "MCPTransportError",
    "ProcessFactory",
    "SSETransport",
    "StdioTransport",
    "Transport",
]
