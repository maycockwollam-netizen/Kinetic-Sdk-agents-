"""Run feedback endpoint (POST /runs/<id>/feedback)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.server.app import MAX_FEEDBACK_COMMENT
from kinetic_sdk.testing import MockLLMClient, text_response


def _factory(session_id: str | None = None) -> Agent:
    return Agent(llm=MockLLMClient([text_response("ok")] * 5), permission_policy=PermissivePolicy())


def _post(url: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())


def _get(url: str, token: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


@pytest.fixture()
def server():
    value = AgentServer(_factory, port=0)
    value.start_in_thread()
    yield value
    value.shutdown()


def _base(server: AgentServer) -> str:
    return f"http://127.0.0.1:{server.port}"


def test_feedback_recorded_and_mirrored_on_run(server):
    record = server._run_agent("hi", stream=False, output_schema=None)
    run_id = record["run_id"]
    status, body = _post(f"{_base(server)}/runs/{run_id}/feedback", {"rating": "up"})
    assert status == 200
    assert body["ok"] is True
    assert body["run_id"] == run_id
    assert body["feedback"]["rating"] == "up"
    # Mirrored onto GET /runs/<id>
    got = _get(f"{_base(server)}/runs/{run_id}")
    assert got["feedback"]["rating"] == "up"


def test_feedback_down_rating_replaces(server):
    run_id = server._run_agent("hi", stream=False, output_schema=None)["run_id"]
    _post(f"{_base(server)}/runs/{run_id}/feedback", {"rating": "up"})
    _, body = _post(f"{_base(server)}/runs/{run_id}/feedback", {"rating": "down"})
    assert body["feedback"]["rating"] == "down"
    assert _get(f"{_base(server)}/runs/{run_id}")["feedback"]["rating"] == "down"


def test_feedback_rejects_bad_rating(server):
    run_id = server._run_agent("hi", stream=False, output_schema=None)["run_id"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{_base(server)}/runs/{run_id}/feedback", {"rating": "meh"})
    assert exc.value.code == 400


def test_feedback_unknown_run_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{_base(server)}/runs/does-not-exist/feedback", {"rating": "up"})
    assert exc.value.code == 404


def test_feedback_comment_is_capped(server):
    run_id = server._run_agent("hi", stream=False, output_schema=None)["run_id"]
    _, body = _post(
        f"{_base(server)}/runs/{run_id}/feedback",
        {"rating": "down", "comment": "x" * (MAX_FEEDBACK_COMMENT + 500)},
    )
    assert len(body["feedback"]["comment"]) == MAX_FEEDBACK_COMMENT


def test_feedback_requires_token():
    value = AgentServer(_factory, port=0, token="sekret")
    value.start_in_thread()
    try:
        run_id = value._run_agent("hi", stream=False, output_schema=None)["run_id"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(f"{_base(value)}/runs/{run_id}/feedback", {"rating": "up"})
        assert exc.value.code == 401
        status, body = _post(f"{_base(value)}/runs/{run_id}/feedback", {"rating": "up"}, token="sekret")
        assert status == 200
        assert body["feedback"]["rating"] == "up"
    finally:
        value.shutdown()