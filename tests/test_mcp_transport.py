"""Tests for kinetic_sdk.mcp.transport (StdioTransport + SSETransport)."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time

import pytest

from kinetic_sdk.mcp.protocol import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)
from kinetic_sdk.mcp.transport import (
    MCPTimeoutError,
    MCPTransportError,
    SSETransport,
    StdioTransport,
)

from ._mcp_fakes import FAKE_MCP_SERVER_SCRIPT, FakeSSEMCPServer


@pytest.fixture()
def fake_server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_MCP_SERVER_SCRIPT)
    return str(path)


def _spawn_fake_server(fake_server_path: str) -> StdioTransport:
    return StdioTransport(command=sys.executable, args=[fake_server_path])


class TestStdioTransport:
    def test_send_receive_round_trip(self, fake_server_path: str) -> None:
        with _spawn_fake_server(fake_server_path) as transport:
            transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
            msg = transport.receive(timeout=5)
            assert isinstance(msg, JsonRpcResponse)
            assert msg.id == 1
            assert msg.result["serverInfo"]["name"] == "fake-mcp"

    def test_notification_gets_no_response(self, fake_server_path: str) -> None:
        with _spawn_fake_server(fake_server_path) as transport:
            transport.send(JsonRpcNotification(method="notifications/initialized"))
            with pytest.raises(MCPTimeoutError):
                transport.receive(timeout=0.3)

    def test_receive_timeout_raises(self, fake_server_path: str) -> None:
        # Server only answers real requests; an unknown-but-unhandled pause is
        # simulated by simply not sending anything before receive().
        with _spawn_fake_server(fake_server_path) as transport:
            with pytest.raises(MCPTimeoutError):
                transport.receive(timeout=0.2)

    def test_dead_subprocess_detected_at_receive(self, tmp_path) -> None:
        script = tmp_path / "die.py"
        script.write_text("import sys; sys.stderr.write('fatal boom\\n'); sys.exit(3)")
        transport = StdioTransport(command=sys.executable, args=[str(script)])
        try:
            with pytest.raises(MCPTransportError, match="code 3"):
                transport.receive(timeout=5)
            # Every subsequent receive must also fail fast, not block.
            with pytest.raises(MCPTransportError):
                transport.receive(timeout=5)
        finally:
            transport.close()

    def test_subprocess_dying_mid_conversation(self, fake_server_path: str) -> None:
        with _spawn_fake_server(fake_server_path) as transport:
            transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
            transport.receive(timeout=5)
            transport.send(JsonRpcNotification(method="notifications/initialized"))
            transport.send(
                JsonRpcRequest(id=2, method="tools/call", params={"name": "crash", "arguments": {}})
            )
            with pytest.raises(MCPTransportError, match="code 3"):
                transport.receive(timeout=5)

    def test_send_after_close_raises(self, fake_server_path: str) -> None:
        transport = _spawn_fake_server(fake_server_path)
        transport.close()
        with pytest.raises(MCPTransportError, match="closed"):
            transport.send(JsonRpcNotification(method="ping"))

    def test_close_is_idempotent_and_kills_process(self, fake_server_path: str) -> None:
        transport = _spawn_fake_server(fake_server_path)
        proc = transport._proc
        transport.close()
        transport.close()  # second close must be a no-op
        assert proc is not None
        assert proc.poll() is not None  # terminated

    def test_env_replacement_passed_to_factory(self) -> None:
        captured: dict = {}

        class _FakeProc:
            def __init__(self) -> None:
                self._r, self._w = os.pipe()
                self.stdin = os.fdopen(self._w, "wb")
                self.stdout = os.fdopen(os.dup(self._r), "rb")
                self.stderr = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        def factory(argv, env):
            captured["argv"] = argv
            captured["env"] = env
            return _FakeProc()

        transport = StdioTransport(
            command="fake-cmd",
            args=["--x", "1"],
            env={"A": "B"},
            process_factory=factory,
        )
        transport.close()
        assert captured["argv"] == ["fake-cmd", "--x", "1"]
        assert captured["env"] == {"A": "B"}

    def test_attach_mode_round_trip_over_pipes(self) -> None:
        # attach() is the server-side entry point; emulate a peer over pipes.
        to_server_r, to_server_w = os.pipe()
        from_server_r, from_server_w = os.pipe()
        transport = StdioTransport.attach(
            reader=os.fdopen(to_server_r, "rb"), writer=os.fdopen(from_server_w, "wb")
        )
        try:
            with os.fdopen(to_server_w, "wb") as peer_in:
                peer_in.write(b'{"jsonrpc": "2.0", "id": 9, "method": "ping"}\n')
            msg = transport.receive(timeout=5)
            assert isinstance(msg, JsonRpcRequest)
            assert msg.id == 9
            transport.send(JsonRpcResponse(id=9, result={"pong": True}))
            with os.fdopen(from_server_r, "rb") as peer_out:
                line = peer_out.readline()
            assert b'"pong": true' in line
        finally:
            transport.close()

    def test_attach_eof_raises_transport_error(self) -> None:
        r, w = os.pipe()
        os.close(w)  # peer gone immediately
        transport = StdioTransport.attach(reader=os.fdopen(r, "rb"), writer=open(os.devnull, "wb"))
        try:
            with pytest.raises(MCPTransportError):
                transport.receive(timeout=5)
        finally:
            transport.close()

    def test_context_manager_closes_on_exception(self, fake_server_path: str) -> None:
        with pytest.raises(RuntimeError, match="user error"):
            with _spawn_fake_server(fake_server_path) as transport:
                proc = transport._proc
                raise RuntimeError("user error")
        assert proc is not None
        assert proc.poll() is not None

    def test_del_cleans_up_orphan_subprocess(self, fake_server_path: str) -> None:
        transport = _spawn_fake_server(fake_server_path)
        proc = transport._proc
        assert proc is not None
        pid = proc.pid
        del transport
        gc.collect()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return  # process is gone: cleanup worked
            time.sleep(0.05)
        pytest.fail(f"subprocess {pid} still alive after transport was dropped")


class TestSSETransport:
    def test_full_handshake_and_call(self) -> None:
        server = FakeSSEMCPServer()
        try:
            with SSETransport(server.url, connect_timeout=2, read_timeout=2) as transport:
                transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
                msg = transport.receive(timeout=2)
                assert isinstance(msg, JsonRpcResponse)
                assert msg.result["serverInfo"]["name"] == "fake-mcp-sse"

                transport.send(JsonRpcNotification(method="notifications/initialized"))
                transport.send(
                    JsonRpcRequest(
                        id=2,
                        method="tools/call",
                        params={"name": "echo", "arguments": {"text": "hello"}},
                    )
                )
                msg = transport.receive(timeout=2)
                assert isinstance(msg, JsonRpcResponse)
                assert msg.id == 2
                assert msg.result["content"][0]["text"] == "hello"
        finally:
            server.close()

    def test_headers_sent_on_get_and_post(self) -> None:
        # A server that rejects requests without the auth header.
        server = FakeSSEMCPServer()
        try:
            transport = SSETransport(
                server.url,
                headers={"Authorization": "Bearer test-token"},
                connect_timeout=2,
                read_timeout=2,
            )
            transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
            msg = transport.receive(timeout=2)
            assert isinstance(msg, JsonRpcResponse)
            transport.close()
        finally:
            server.close()

    def test_silent_server_times_out_on_open(self) -> None:
        server = FakeSSEMCPServer(behavior="silent")
        try:
            transport = SSETransport(server.url, connect_timeout=0.3, read_timeout=0.3)
            with pytest.raises(MCPTimeoutError):
                transport.receive(timeout=0.3)
            transport.close()
        finally:
            server.close()

    def test_receive_timeout_after_endpoint(self) -> None:
        server = FakeSSEMCPServer(behavior="quiet")
        try:
            transport = SSETransport(server.url, connect_timeout=2, read_timeout=0.3)
            transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
            with pytest.raises(MCPTimeoutError):
                transport.receive(timeout=0.3)
            transport.close()
        finally:
            server.close()

    def test_unreachable_server_raises_transport_error(self) -> None:
        # Port 1 on localhost is never a listening MCP server.
        transport = SSETransport("http://127.0.0.1:1/sse", connect_timeout=0.5)
        with pytest.raises(MCPTransportError, match="Cannot connect"):
            transport.send(JsonRpcNotification(method="ping"))

    def test_bad_scheme_rejected(self) -> None:
        transport = SSETransport("ftp://example.com/sse")
        with pytest.raises(MCPTransportError, match="http"):
            transport.send(JsonRpcNotification(method="ping"))

    def test_close_is_idempotent(self) -> None:
        server = FakeSSEMCPServer()
        try:
            transport = SSETransport(server.url, connect_timeout=2)
            transport.send(JsonRpcRequest(id=1, method="initialize", params={}))
            transport.receive(timeout=2)
            transport.close()
            transport.close()
            with pytest.raises(MCPTransportError, match="closed"):
                transport.send(JsonRpcNotification(method="ping"))
        finally:
            server.close()
