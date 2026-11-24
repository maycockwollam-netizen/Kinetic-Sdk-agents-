"""Session, OpenAI gateway, and WebSocket coverage for AgentServer."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time
import urllib.error
import urllib.request

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.testing import MockLLMClient, text_response


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def _factory(session_id: str | None = None) -> Agent:
    def reply(messages, _tools, _system):
        users = [item["content"] for item in messages if item["role"] == "user" and isinstance(item["content"], str)]
        return text_response("seen: " + users[-1])

    return Agent(llm=MockLLMClient([reply] * 10), permission_policy=PermissivePolicy())


@pytest.fixture()
def server():
    value = AgentServer(_factory, port=0, session_ttl=0.05)
    value.start_in_thread()
    yield value
    value.shutdown()


def test_runs_reuse_session_conversation(server):
    root = f"http://127.0.0.1:{server.port}"
    _post(root + "/runs", {"message": "first", "session_id": "chat-a"})
    _post(root + "/runs", {"message": "second", "session_id": "chat-a"})
    assert [message["content"] for message in server._sessions["chat-a"].state.messages if message["role"] == "user"] == ["first", "second"]

    time.sleep(0.06)
    _post(root + "/runs", {"message": "third", "session_id": "chat-a"})
    assert [message["content"] for message in server._sessions["chat-a"].state.messages if message["role"] == "user"] == ["third"]


def test_server_evicts_old_run_records_at_configured_bound():
    value = AgentServer(_factory, port=0, max_runs=2)
    try:
        first = value._run_agent("one", stream=False, output_schema=None)
        second = value._run_agent("two", stream=False, output_schema=None)
        third = value._run_agent("three", stream=False, output_schema=None)
        assert first["run_id"] not in value._runs
        assert list(value._runs) == [second["run_id"], third["run_id"]]
    finally:
        value._httpd.server_close()


def test_server_rejects_nonpositive_resource_bounds():
    with pytest.raises(ValueError, match="max_runs"):
        AgentServer(_factory, port=0, max_runs=0)
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        AgentServer(_factory, port=0, max_concurrent_requests=0)


def test_different_sessions_run_concurrently():
    """A long run in one session must not block a different session's run."""

    def timing_factory(session_id: str | None = None) -> Agent:
        if session_id == "slow":
            def slow(messages, _tools, _system):
                time.sleep(0.3)
                return text_response("slow done")

            return Agent(llm=MockLLMClient([slow]), permission_policy=PermissivePolicy())

        def fast(messages, _tools, _system):
            return text_response("fast done")

        return Agent(llm=MockLLMClient([fast]), permission_policy=PermissivePolicy())

    import threading

    value = AgentServer(timing_factory, port=0)
    try:
        box: dict[str, object] = {}
        thread = threading.Thread(
            target=lambda: box.setdefault("slow", value._run_agent("s", stream=False, output_schema=None, session_id="slow"))
        )
        thread.start()
        time.sleep(0.02)  # let the slow session start and hold its lock
        started = time.monotonic()
        fast_record = value._run_agent("f", stream=False, output_schema=None, session_id="fast")
        elapsed = time.monotonic() - started
        thread.join()
        assert fast_record["status"] == "completed"
        assert elapsed < 0.2, f"fast session blocked by slow session for {elapsed:.3f}s"
    finally:
        value._httpd.server_close()


def test_same_session_serialises_runs():
    """Two runs on ONE cached agent must still be serialised (no race)."""

    def slow_factory(session_id: str | None = None) -> Agent:
        def slow(messages, _tools, _system):
            time.sleep(0.1)
            return text_response("done")

        return Agent(llm=MockLLMClient([slow] * 10), permission_policy=PermissivePolicy())

    value = AgentServer(slow_factory, port=0)
    try:
        started = time.monotonic()
        first = value._run_agent("a", stream=False, output_schema=None, session_id="same")
        second = value._run_agent("b", stream=False, output_schema=None, session_id="same")
        elapsed = time.monotonic() - started
        assert first["status"] == "completed" and second["status"] == "completed"
        assert elapsed >= 0.19, f"same-session runs overlapped: {elapsed:.3f}s"
    finally:
        value._httpd.server_close()


def test_openai_completion_uses_model_as_session_id(server):
    root = f"http://127.0.0.1:{server.port}"
    result = _post(root + "/v1/chat/completions", {"model": "openai-session", "messages": [{"role": "user", "content": "hello"}], "stream": False})
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "seen: hello"
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(root + "/v1/chat/completions", {"model": "x", "messages": [], "stream": True})
    assert error.value.code == 400


def _masked_frame(payload: str) -> bytes:
    data, mask = payload.encode(), os.urandom(4)
    assert len(data) < 126
    return bytes([0x81, 0x80 | len(data)]) + mask + bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))


def _read_frame(sock: socket.socket) -> str:
    head = sock.recv(2)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    return sock.recv(length).decode()


def _read_raw_frame(sock: socket.socket) -> tuple[int, bytes]:
    head = sock.recv(2)
    assert len(head) == 2
    opcode = head[0] & 0x0F
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", sock.recv(8))[0]
    return opcode, sock.recv(length)


def _ws_frame(opcode: int, payload: bytes, *, fin: bool = True) -> bytes:
    """Build a masked client frame (opcode, payload, FIN)."""
    mask = os.urandom(4)
    first = (0x80 if fin else 0) | opcode
    if len(payload) < 126:
        header = bytes([first, 0x80 | len(payload)])
    elif len(payload) <= 0xFFFF:
        header = bytes([first, 0x80 | 126]) + struct.pack("!H", len(payload))
    else:
        header = bytes([first, 0x80 | 127]) + struct.pack("!Q", len(payload))
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    return header + mask + masked


def _upgrade(sock: socket.socket) -> None:
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((
        "GET /runs/stream HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Key: {key}\r\n\r\n"
    ).encode())
    assert b"101 Switching Protocols" in sock.recv(1024)


def _run_until_completed(sock: socket.socket) -> list[dict]:
    events = []
    while True:
        event = json.loads(_read_frame(sock))
        events.append(event)
        if event.get("type") == "run.completed":
            return events


def test_websocket_streams_agent_events(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((
            "GET /runs/stream HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {key}\r\n\r\n"
        ).encode())
        assert b"101 Switching Protocols" in sock.recv(1024)
        sock.sendall(_masked_frame(json.dumps({"message": "socket"})))
        events = []
        while True:
            event = _read_frame(sock)
            events.append(event)
            if json.loads(event).get("type") == "run.completed":
                break
        assert any(json.loads(event).get("type") == "agent.run_started" for event in events)
        assert json.loads(events[-1]) == {"type": "run.completed", "final": "seen: socket"}
    finally:
        sock.close()


def test_websocket_accepts_multiple_messages_on_one_connection(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        _upgrade(sock)
        sock.sendall(_masked_frame(json.dumps({"message": "one"})))
        first = _run_until_completed(sock)
        assert first[-1] == {"type": "run.completed", "final": "seen: one"}
        sock.sendall(_masked_frame(json.dumps({"message": "two"})))
        second = _run_until_completed(sock)
        assert second[-1] == {"type": "run.completed", "final": "seen: two"}
    finally:
        sock.close()


def test_websocket_replies_to_ping_with_pong(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        _upgrade(sock)
        sock.sendall(_ws_frame(0x9, b"hi"))  # ping
        opcode, payload = _read_raw_frame(sock)
        assert opcode == 0xA
        assert payload == b"hi"
        # connection must still be usable after the ping/pong exchange
        sock.sendall(_masked_frame(json.dumps({"message": "socket"})))
        events = _run_until_completed(sock)
        assert events[-1] == {"type": "run.completed", "final": "seen: socket"}
    finally:
        sock.close()


def test_websocket_reassembles_fragmented_text(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        _upgrade(sock)
        payload = json.dumps({"message": "socket"}).encode()
        sock.sendall(_ws_frame(0x1, payload[:5], fin=False))      # text start
        sock.sendall(_ws_frame(0x0, payload[5:10], fin=False))    # continuation
        sock.sendall(_ws_frame(0x0, payload[10:], fin=True))      # final continuation
        events = _run_until_completed(sock)
        assert events[-1] == {"type": "run.completed", "final": "seen: socket"}
    finally:
        sock.close()


def test_websocket_close_handshake(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=2)
    try:
        _upgrade(sock)
        sock.sendall(_ws_frame(0x8, struct.pack("!H", 1000)))
        opcode, payload = _read_raw_frame(sock)
        assert opcode == 0x8
        assert struct.unpack("!H", payload[:2])[0] == 1000
    finally:
        sock.close()


def test_openai_completion_streams_sse(server):
    root = f"http://127.0.0.1:{server.port}"
    request = urllib.request.Request(
        root + "/v1/chat/completions",
        data=json.dumps({
            "model": "stream-session",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        assert response.headers.get_content_type() == "text/event-stream"
        body = response.read().decode()
    lines = [line for line in body.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    parsed = [json.loads(line[6:]) for line in lines[:-1]]
    assert all(item["object"] == "chat.completion.chunk" for item in parsed)
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
    content = "".join(item["choices"][0]["delta"].get("content", "") for item in parsed)
    assert content == "seen: hi"


def test_session_store_persists_runs_across_restart(tmp_path):
    store = tmp_path / "store"
    first = AgentServer(_factory, port=0, session_store_dir=store)
    try:
        record = first._run_agent("hello", stream=False, output_schema=None)
        run_id = record["run_id"]
    finally:
        first._httpd.server_close()
    # Simulate a restart: a fresh server over the same directory reloads runs.
    second = AgentServer(_factory, port=0, session_store_dir=store)
    try:
        assert run_id in second._runs
        assert second._runs[run_id]["status"] == "completed"
        assert second._runs[run_id]["final"] == "seen: hello"
    finally:
        second._httpd.server_close()


def test_runs_stream_returns_sse_event_stream(server):
    root = f"http://127.0.0.1:{server.port}"
    request = urllib.request.Request(
        root + "/runs",
        data=json.dumps({"message": "hi", "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        body = response.read().decode("utf-8")

    assert "event: agent.run_started\n" in body
    assert "event: agent.run_finished\n" in body
    assert "event: done\n" in body  # final RunRecord payload
    assert '"final"' in body
    assert '"status": "completed"' in body
