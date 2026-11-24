"""MCPServerRegistry.reconnect: dead transports heal by re-handshaking."""

from __future__ import annotations

import sys

import pytest

from kinetic_sdk.mcp.registry import MCPServerConfig, MCPServerRegistry

from ._mcp_fakes import FAKE_MCP_SERVER_SCRIPT


@pytest.fixture()
def fake_server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_MCP_SERVER_SCRIPT)
    return str(path)


def test_reconnect_replaces_cached_client(fake_server_path):
    registry = MCPServerRegistry()
    registry.register(
        "fake",
        MCPServerConfig.stdio(command=sys.executable, args=[fake_server_path]),
    )
    try:
        first = registry.connect("fake")
        assert registry.is_connected("fake")
        second = registry.reconnect("fake")
        assert second is not first
        # The fresh client works: the fake server answers initialize again.
        assert second.initialized
        tools = second.list_tools()
        assert tools
    finally:
        registry.close_all()


def test_reconnect_unknown_server_raises():
    registry = MCPServerRegistry()
    with pytest.raises(KeyError):
        registry.reconnect("nope")


def test_reconnect_without_prior_connect(fake_server_path):
    registry = MCPServerRegistry()
    registry.register(
        "fake",
        MCPServerConfig.stdio(command=sys.executable, args=[fake_server_path]),
    )
    try:
        client = registry.reconnect("fake")  # reconnect == fresh connect here
        assert client.initialized
    finally:
        registry.close_all()


def test_adapter_auto_reconnects_after_transport_death(fake_server_path):
    """A tool call on a DEAD transport heals via one proactive reconnect."""
    registry = MCPServerRegistry()
    registry.register(
        "fake",
        MCPServerConfig.stdio(command=sys.executable, args=[fake_server_path]),
    )
    try:
        tools = registry.get_tools_as_kinetic_tools("fake")
        by_name = {tool.name: tool for tool in tools}
        echo = by_name["fake.echo"]
        crash = by_name["fake.crash"]
        # Healthy call works and the cached client is shared.
        assert echo.execute(text="ok").output == "ok"
        # The server dies mid-session: the crash tool exits the subprocess
        # without answering, killing the shared transport.
        assert crash.execute().is_error
        # The next echo AUTO-reconnects (fresh subprocess) and retries.
        healed = echo.execute(text="after-crash")
        assert not healed.is_error, healed.error
        assert healed.output == "after-crash"
    finally:
        registry.close_all()


def test_adapter_timeout_does_not_trigger_reconnect(fake_server_path):
    """A mere timeout must NOT re-spawn the server (a live-but-slow peer)."""
    from kinetic_sdk.mcp.adapter import MCPToolAdapter
    from kinetic_sdk.mcp.transport import MCPTimeoutError

    registry = MCPServerRegistry()
    registry.register(
        "fake",
        MCPServerConfig.stdio(command=sys.executable, args=[fake_server_path]),
    )
    try:
        client = registry.connect("fake")

        def _timeout(*_args: object, **_kwargs: object) -> object:
            raise MCPTimeoutError("slow")

        client.call_tool = _timeout  # type: ignore[method-assign]
        calls = {"n": 0}

        def reconnect() -> object:
            calls["n"] += 1
            return registry.reconnect("fake")

        adapter = MCPToolAdapter(client, "echo", "fake", reconnector=reconnect)  # type: ignore[arg-type]
        result = adapter.execute(text="hi")
        assert result.is_error
        assert calls["n"] == 0  # timeout must NOT trigger a reconnect
    finally:
        registry.close_all()
