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
