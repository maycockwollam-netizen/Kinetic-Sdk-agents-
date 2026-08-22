"""Tests for kinetic_sdk.mcp.registry (MCPServerConfig + MCPServerRegistry)."""

from __future__ import annotations

import sys

import pytest

from kinetic_sdk.mcp.client import MCPClient
from kinetic_sdk.mcp.registry import MCPServerConfig, MCPServerRegistry
from kinetic_sdk.mcp.transport import MCPTransportError
from kinetic_sdk.secret.provider import DictSecretProvider
from kinetic_sdk.secret.registry import SecretRegistry
from kinetic_sdk.secret.value import SecretValue
from kinetic_sdk.tool.base import Tool

from ._mcp_fakes import FAKE_MCP_SERVER_SCRIPT, FakeSSEMCPServer


@pytest.fixture()
def fake_server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_MCP_SERVER_SCRIPT)
    return str(path)


def _stdio_config(fake_server_path: str, **kwargs) -> MCPServerConfig:
    return MCPServerConfig.stdio(
        command=sys.executable, args=[fake_server_path], **kwargs
    )


class TestConfig:
    def test_stdio_flavour(self) -> None:
        config = MCPServerConfig.stdio("unity-mcp", args=["--x"], env={"A": "B"})
        assert config.transport_kind == "stdio"
        assert config.command == "unity-mcp"
        assert config.args == ["--x"]

    def test_sse_flavour(self) -> None:
        config = MCPServerConfig.sse("http://localhost:9/sse", headers={"H": "v"})
        assert config.transport_kind == "sse"
        assert config.url == "http://localhost:9/sse"

    def test_exactly_one_flavour_required(self) -> None:
        with pytest.raises(ValueError, match="XOR"):
            MCPServerConfig()  # neither
        with pytest.raises(ValueError, match="XOR"):
            MCPServerConfig(command="x", url="http://y")  # both

    def test_secret_value_in_config_not_leaked_by_repr(self) -> None:
        secret = SecretValue("super-secret-token-12345")
        config = MCPServerConfig.sse("http://x/sse", headers={"Authorization": secret})
        assert "super-secret-token-12345" not in repr(config)


class TestRegistryBasics:
    def test_register_and_list_names(self, fake_server_path: str) -> None:
        registry = MCPServerRegistry()
        registry.register("a", _stdio_config(fake_server_path))
        registry.register("b", _stdio_config(fake_server_path))
        assert registry.registered_names() == ["a", "b"]
        assert not registry.is_connected("a")

    def test_duplicate_registration_rejected(self, fake_server_path: str) -> None:
        registry = MCPServerRegistry()
        registry.register("a", _stdio_config(fake_server_path))
        with pytest.raises(ValueError, match="already registered"):
            registry.register("a", _stdio_config(fake_server_path))

    def test_empty_name_rejected(self, fake_server_path: str) -> None:
        with pytest.raises(ValueError):
            MCPServerRegistry().register("", _stdio_config(fake_server_path))

    def test_connect_unknown_name_raises_with_known_names(self) -> None:
        registry = MCPServerRegistry()
        with pytest.raises(KeyError, match="ghost"):
            registry.connect("ghost")


class TestConnect:
    def test_connect_initializes_and_caches_client(self, fake_server_path: str) -> None:
        with MCPServerRegistry() as registry:
            registry.register("fake", _stdio_config(fake_server_path))
            client = registry.connect("fake")
            assert isinstance(client, MCPClient)
            assert client.initialized
            assert registry.is_connected("fake")
            assert registry.connect("fake") is client  # cached

    def test_failed_handshake_leaves_no_orphan(self, tmp_path) -> None:
        script = tmp_path / "die.py"
        script.write_text("import sys; sys.exit(1)")
        registry = MCPServerRegistry()
        registry.register("dead", _stdio_config(str(script)))
        with pytest.raises(MCPTransportError):
            registry.connect("dead")
        assert not registry.is_connected("dead")
        # A retry spawns a FRESH attempt (the failed one was cleaned up).
        with pytest.raises(MCPTransportError):
            registry.connect("dead")

    def test_close_then_reconnect_spawns_new_client(self, fake_server_path: str) -> None:
        registry = MCPServerRegistry()
        registry.register("fake", _stdio_config(fake_server_path))
        first = registry.connect("fake")
        registry.close("fake")
        assert not registry.is_connected("fake")
        second = registry.connect("fake")
        assert second is not first
        registry.close_all()

    def test_close_is_idempotent(self, fake_server_path: str) -> None:
        registry = MCPServerRegistry()
        registry.register("fake", _stdio_config(fake_server_path))
        registry.connect("fake")
        registry.close("fake")
        registry.close("fake")
        registry.close("never-registered")


class TestTools:
    def test_get_tools_as_kinetic_tools(self, fake_server_path: str) -> None:
        with MCPServerRegistry() as registry:
            registry.register("fake", _stdio_config(fake_server_path))
            tools = registry.get_tools_as_kinetic_tools("fake")
            assert all(isinstance(t, Tool) for t in tools)
            assert {t.name for t in tools} == {"fake.echo", "fake.fail", "fake.crash"}
            echo = next(t for t in tools if t.name == "fake.echo")
            result = echo.execute(text="hello")
            assert not result.is_error
            assert result.output == "hello"

    def test_two_servers_same_tool_names_do_not_collide(self, fake_server_path: str) -> None:
        with MCPServerRegistry() as registry:
            registry.register("serverA", _stdio_config(fake_server_path))
            registry.register("serverB", _stdio_config(fake_server_path))
            tools = registry.get_tools_as_kinetic_tools(
                "serverA"
            ) + registry.get_tools_as_kinetic_tools("serverB")
            names = [t.name for t in tools]
            assert len(names) == len(set(names))  # no duplicates
            assert "serverA.echo" in names
            assert "serverB.echo" in names
            # Both are independently callable against their own server.
            by_name = {t.name: t for t in tools}
            assert by_name["serverA.echo"].execute(text="A").output == "A"
            assert by_name["serverB.echo"].execute(text="B").output == "B"


class TestCredentials:
    def test_secret_value_env_revealed_only_to_subprocess(self, tmp_path) -> None:
        # Server reports back the env var it saw inside serverInfo.name, so
        # the test can verify the plaintext reached the child process.
        script = tmp_path / "env_server.py"
        script.write_text(
            "import json, os, sys\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)\n"
            "    if msg.get('method') == 'initialize':\n"
            "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'],\n"
            "            'result': {'protocolVersion': '2025-03-26', 'capabilities': {},\n"
            "            'serverInfo': {'name': os.environ.get('MCP_TEST_SECRET', ''), 'version': '1'}}}) + '\\n')\n"
            "        sys.stdout.flush()\n"
        )
        secrets = SecretRegistry([DictSecretProvider({"MCP_TEST_SECRET": "s3cr3t-value"})])
        with MCPServerRegistry(secrets=secrets) as registry:
            registry.register(
                "env",
                MCPServerConfig.stdio(
                    command=sys.executable,
                    args=[str(script)],
                    env={"MCP_TEST_SECRET": secrets.resolve("MCP_TEST_SECRET")},
                ),
            )
            client = registry.connect("env")
            assert client.server_info["serverInfo"]["name"] == "s3cr3t-value"

    def test_sse_headers_secret_revealed_on_the_wire(self) -> None:
        server = FakeSSEMCPServer()
        try:
            secrets = SecretRegistry([DictSecretProvider({"MCP_TOKEN": "tok-abc-123"})])
            with MCPServerRegistry(secrets=secrets) as registry:
                registry.register(
                    "remote",
                    MCPServerConfig.sse(
                        url=server.url,
                        headers={"Authorization": secrets.resolve("MCP_TOKEN")},
                    ),
                )
                client = registry.connect("remote")
                assert client.initialized
                assert "tok-abc-123" in server.received_auth_headers
        finally:
            server.close()

    def test_plain_string_values_pass_through(self, fake_server_path: str) -> None:
        with MCPServerRegistry() as registry:
            registry.register(
                "fake",
                _stdio_config(fake_server_path, env={"PLAIN": "value"}),
            )
            assert registry.connect("fake").initialized


class TestSSEConnect:
    def test_connect_over_sse(self) -> None:
        server = FakeSSEMCPServer()
        try:
            with MCPServerRegistry() as registry:
                registry.register("remote", MCPServerConfig.sse(url=server.url))
                tools = registry.get_tools_as_kinetic_tools("remote")
                assert [t.name for t in tools] == ["remote.echo"]
                assert tools[0].execute(text="hi").output == "hi"
        finally:
            server.close()
