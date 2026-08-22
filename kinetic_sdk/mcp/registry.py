"""Client-side MCP server registry (Stage 4, MCP module).

An agent rarely talks to just one MCP server — a game-dev agent might drive
Unity AND a filesystem server AND GitHub at once. :class:`MCPServerRegistry`
holds the per-server configs, connects lazily (``register`` only stores the
config; the subprocess/connection is spawned on first ``connect``), and
hands out ready-to-use Kinetic tools via :meth:`get_tools_as_kinetic_tools`.

Credential lifecycle follows the same rule as :mod:`kinetic_sdk.git`:
``env`` / ``headers`` values may be plain strings (non-secret settings) or
:class:`~kinetic_sdk.secret.value.SecretValue` objects resolved from a
:class:`~kinetic_sdk.secret.registry.SecretRegistry`. The plaintext is only
revealed at the moment the transport is built, never stored, and this
module never reads ``os.environ`` directly::

    secrets = SecretRegistry()  # env-based
    registry = MCPServerRegistry(secrets=secrets)
    registry.register(
        "github",
        MCPServerConfig.sse(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {secrets.resolve('GITHUB_TOKEN').reveal()}"},
        ),
    )

or, keeping the value wrapped until the last moment (preferred)::

    registry.register(
        "unity",
        MCPServerConfig.stdio(
            command="unity-mcp",
            args=["--project", "./Game"],
            env={"UNITY_API_KEY": secrets.resolve("UNITY_API_KEY")},
        ),
    )
    tools = registry.get_tools_as_kinetic_tools("unity")  # names: unity.*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kinetic_sdk.mcp.adapter import MCPToolAdapter
from kinetic_sdk.mcp.client import MCPClient
from kinetic_sdk.mcp.transport import SSETransport, StdioTransport, Transport
from kinetic_sdk.secret.registry import SecretRegistry
from kinetic_sdk.secret.value import SecretValue
from kinetic_sdk.tool.base import Tool

#: A config value: either a literal string or a wrapped secret.
ConfigValue = str | SecretValue


def _reveal_map(mapping: dict[str, ConfigValue] | None) -> dict[str, str] | None:
    """Reveal any SecretValue entries, at transport-build time only."""
    if mapping is None:
        return None
    return {
        key: value.reveal() if isinstance(value, SecretValue) else value
        for key, value in mapping.items()
    }


@dataclass
class MCPServerConfig:
    """Configuration for ONE MCP server (exactly one transport flavour).

    Use the :meth:`stdio` / :meth:`sse` constructors rather than the raw
    dataclass — they make the flavour explicit.
    """

    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, ConfigValue] | None = None
    url: str | None = None
    headers: dict[str, ConfigValue] | None = None
    init_timeout: float = 10.0
    request_timeout: float = 30.0

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, ConfigValue] | None = None,
        **kwargs: Any,
    ) -> "MCPServerConfig":
        """A local server spawned as a subprocess (the common MCP case)."""
        return cls(command=command, args=list(args or []), env=env, **kwargs)

    @classmethod
    def sse(
        cls,
        url: str,
        headers: dict[str, ConfigValue] | None = None,
        **kwargs: Any,
    ) -> "MCPServerConfig":
        """A remote server reached over HTTP/SSE."""
        return cls(url=url, headers=headers, **kwargs)

    def __post_init__(self) -> None:
        if (self.command is None) == (self.url is None):
            raise ValueError(
                "MCPServerConfig needs exactly one transport flavour: "
                "'command' (stdio) XOR 'url' (SSE)"
            )

    @property
    def transport_kind(self) -> str:
        return "stdio" if self.command is not None else "sse"


class MCPServerRegistry:
    """Registry of named MCP servers; connects lazily, caches clients.

    Args:
        secrets: Registry used to resolve credentials referenced from server
            configs. Defaults to the environment-based registry. This module
            NEVER reads ``os.environ`` itself — all secret resolution goes
            through this object.
    """

    def __init__(self, secrets: SecretRegistry | None = None) -> None:
        self._secrets = secrets if secrets is not None else SecretRegistry()
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}

    @property
    def secrets(self) -> SecretRegistry:
        return self._secrets

    def register(self, name: str, config: MCPServerConfig) -> None:
        """Store *config* under *name*. Does NOT connect yet."""
        if not name:
            raise ValueError("Server name must be non-empty")
        if name in self._configs:
            raise ValueError(
                f"MCP server {name!r} is already registered — close() and "
                "re-register to replace it"
            )
        self._configs[name] = config

    def registered_names(self) -> list[str]:
        return sorted(self._configs)

    def is_connected(self, name: str) -> bool:
        return name in self._clients

    def connect(self, name: str) -> MCPClient:
        """Connect to *name* (spawning the transport) and run the handshake.

        The connected client is cached; a second call returns the same
        instance. If the handshake fails the half-open transport is closed
        immediately so no orphaned subprocess/connection is left behind.
        """
        if name in self._clients:
            return self._clients[name]
        config = self._configs.get(name)
        if config is None:
            raise KeyError(
                f"No MCP server named {name!r} registered "
                f"(known: {self.registered_names() or 'none'})"
            )
        client = MCPClient(
            self._build_transport(config),
            init_timeout=config.init_timeout,
            request_timeout=config.request_timeout,
        )
        try:
            client.initialize()
        except Exception:
            client.close()  # never leak a half-connected subprocess
            raise
        self._clients[name] = client
        return client

    def get_tools_as_kinetic_tools(self, name: str) -> list[Tool]:
        """Connect (if needed) and wrap every server tool in an adapter.

        Tool names are prefixed with the server name (``"unity.search"``),
        so merging tools from several servers cannot collide.
        """
        client = self.connect(name)
        return [
            MCPToolAdapter.from_mcp_schema(client, name, schema)
            for schema in client.list_tools()
        ]

    def close(self, name: str) -> None:
        """Close one server's client (idempotent; keeps the config)."""
        client = self._clients.pop(name, None)
        if client is not None:
            client.close()

    def close_all(self) -> None:
        for name in list(self._clients):
            self.close(name)

    def __enter__(self) -> "MCPServerRegistry":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close_all()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _build_transport(config: MCPServerConfig) -> Transport:
        if config.transport_kind == "stdio":
            assert config.command is not None
            return StdioTransport(
                command=config.command,
                args=config.args,
                env=_reveal_map(config.env),
            )
        assert config.url is not None
        return SSETransport(
            url=config.url,
            headers=_reveal_map(config.headers),
            connect_timeout=config.init_timeout,
            read_timeout=config.request_timeout,
        )
