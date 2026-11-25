"""Runtime settings API (ServerSettings + GET/POST /settings)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.server import AgentServer, ServerSettings
from kinetic_sdk.testing import MockLLMClient, text_response


def _factory(session_id: str | None = None) -> Agent:
    return Agent(llm=MockLLMClient([text_response("ok")] * 5), permission_policy=PermissivePolicy())


def _get(url: str, token: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
        return json.loads(resp.read())


def _post(url: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


# --- ServerSettings -------------------------------------------------------


def test_public_masks_secret_and_reports_set():
    settings = ServerSettings({"model": "gpt-4o", "api_key": "sk-xyz"}, secret_keys={"api_key"})
    public = settings.public()
    assert public["model"] == "gpt-4o"
    assert "api_key" not in public
    assert public["secrets_set"] == {"api_key": True}
    # snapshot() (server-side) still has the real value
    assert settings.snapshot()["api_key"] == "sk-xyz"


def test_update_ignores_none_and_keeps_secret():
    settings = ServerSettings({"model": "a", "api_key": "k"}, secret_keys={"api_key"})
    public = settings.update({"model": "b", "api_key": None})
    assert public["model"] == "b"
    assert settings.snapshot()["api_key"] == "k"  # None did not wipe it


# --- AgentServer integration ---------------------------------------------


@pytest.fixture()
def server():
    settings = ServerSettings({"model": "gpt-4o", "base_url": "", "api_key": ""}, secret_keys={"api_key"})
    value = AgentServer(_factory, port=0, settings=settings)
    value.start_in_thread()
    yield value
    value.shutdown()


def _base(server: AgentServer) -> str:
    return f"http://127.0.0.1:{server.port}"


def test_get_settings_returns_public_view(server):
    body = _get(f"{_base(server)}/settings")
    assert body["model"] == "gpt-4o"
    assert body["secrets_set"] == {"api_key": False}


def test_post_settings_updates_and_masks_key(server):
    status, body = _post(
        f"{_base(server)}/settings",
        {"model": "claude-x", "base_url": "https://api.example.com", "api_key": "sk-secret"},
    )
    assert status == 200
    assert body["ok"] is True
    assert body["settings"]["model"] == "claude-x"
    assert "api_key" not in body["settings"]
    assert body["settings"]["secrets_set"]["api_key"] is True
    # Persisted on the server, masked on GET.
    got = _get(f"{_base(server)}/settings")
    assert got["base_url"] == "https://api.example.com"
    assert "sk-secret" not in json.dumps(got)


def test_settings_change_clears_cached_sessions(server):
    server._run_agent("hi", stream=False, output_schema=None, session_id="chat-a")
    assert "chat-a" in server._sessions
    server.update_settings({"model": "other"})
    assert "chat-a" not in server._sessions


def test_settings_requires_token():
    settings = ServerSettings({"model": "m"}, secret_keys={"api_key"})
    value = AgentServer(_factory, port=0, token="sekret", settings=settings)
    value.start_in_thread()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{_base(value)}/settings")
        assert exc.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{_base(value)}/settings", {"model": "x"})
        assert exc.value.code == 401
        got = _get(f"{_base(value)}/settings", token="sekret")
        assert got["model"] == "m"
    finally:
        value.shutdown()


def test_settings_endpoints_404_without_settings():
    value = AgentServer(_factory, port=0)
    value.start_in_thread()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"{_base(value)}/settings")
        assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{_base(value)}/settings", {"model": "x"})
        assert exc.value.code == 404
    finally:
        value.shutdown()