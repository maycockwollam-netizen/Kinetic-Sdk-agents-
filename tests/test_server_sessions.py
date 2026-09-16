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
