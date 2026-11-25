"""Static web-UI serving for AgentServer (used by ``kinetic serve``)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.testing import MockLLMClient, text_response


def _factory(session_id: str | None = None) -> Agent:
    def reply(messages, _tools, _system):
        users = [m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
        return text_response("ok:" + users[-1])

    return Agent(llm=MockLLMClient([reply] * 10), permission_policy=PermissivePolicy())


@pytest.fixture()
def static_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html><head><title>Kinetic UI</title></head></html>", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log('hello');", encoding="utf-8")
    (tmp_path / "style.css").write_text("body{color:red;}", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def server(static_dir):
    value = AgentServer(_factory, port=0, static_dir=static_dir)
    value.start_in_thread()
    yield value
    value.shutdown()


def _get(url: str) -> tuple[int, str, dict]:
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
        return response.status, body, dict(response.headers)


def test_serves_index_at_root(server):
    status, body, headers = _get(f"http://127.0.0.1:{server.port}/")
    assert status == 200
    assert "Kinetic UI" in body
    assert headers["Content-Type"].startswith("text/html")


def test_serves_static_assets(server):
    status, body, headers = _get(f"http://127.0.0.1:{server.port}/static/app.js")
    assert status == 200
    assert body == "console.log('hello');"
    assert headers["Content-Type"] == "text/javascript; charset=utf-8"
    status, body, _ = _get(f"http://127.0.0.1:{server.port}/static/style.css")
    assert status == 200
    assert "body{color:red;}" in body


def test_missing_static_file_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"http://127.0.0.1:{server.port}/static/nope.png")
    assert exc.value.code == 404


def test_static_is_served_even_with_bearer_token_enabled(tmp_path):
    (tmp_path / "index.html").write_text("token-ui", encoding="utf-8")
    value = AgentServer(_factory, port=0, static_dir=tmp_path, token="sekret")
    value.start_in_thread()
    try:
        status, body, _ = _get(f"http://127.0.0.1:{value.port}/")
        assert status == 200
        assert body == "token-ui"
        # API stays protected.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"http://127.0.0.1:{value.port}/runs/whatever")
        assert exc.value.code == 401
    finally:
        value.shutdown()


def test_static_path_rejects_traversal(static_dir):
    (static_dir.parent / "secret.txt").write_text("classified", encoding="utf-8")
    server = AgentServer(_factory, port=0, static_dir=static_dir)
    try:
        assert server._static_path("index.html") is not None
        assert server._static_path("app.js") is not None
        # Escaping the root (via ..) is rejected even when the target exists.
        assert server._static_path("../secret.txt") is None
        assert server._static_path("..") is None
        assert server._static_path("../") is None
        # Absolute paths are rejected.
        assert server._static_path(str(static_dir.parent / "secret.txt")) is None
        # Missing files simply 404.
        assert server._static_path("missing.html") is None
    finally:
        server._httpd.server_close()


def test_static_dir_must_exist(tmp_path):
    with pytest.raises(ValueError, match="static_dir"):
        AgentServer(_factory, port=0, static_dir=tmp_path / "absent")


def test_api_still_works_alongside_static(server, static_dir):
    status, body, _ = _get(f"http://127.0.0.1:{server.port}/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"