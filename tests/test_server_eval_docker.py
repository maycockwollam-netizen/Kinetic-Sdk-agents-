"""AgentServer REST API, EvalRunner, docker sandbox wrappers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.eval import (
    EvalCase,
    EvalRunner,
    exact_match,
    no_tool_failures,
    tool_called,
)
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.terminal.docker import (
    SandboxWrapperError,
    docker_exec_wrapper,
    docker_run_wrapper,
)
from kinetic_sdk.terminal.tool import TerminalTool
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from tests._helpers import EchoTool

# --- Server -------------------------------------------------------------

BASE = "http://127.0.0.1"


def _factory() -> Agent:
    return Agent(
        llm=MockLLMClient([text_response("pong")]),
        permission_policy=PermissivePolicy(),
    )


@pytest.fixture()
def server():
    srv = AgentServer(_factory, port=0)
    srv.start_in_thread()
    yield srv
    srv.shutdown()


def _post(url, payload, token=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(url, token=None):
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def test_server_health(server):
    data = _get(f"{BASE}:{server.port}/health")
    assert data["status"] == "ok"


def test_server_runs_roundtrip(server):
    url = f"{BASE}:{server.port}"
    result = _post(url + "/runs", {"message": "ping"})
    assert result["status"] == "completed" and result["final"] == "pong"
    assert "usage" in result
    fetched = _get(url + f"/runs/{result['run_id']}")
    assert fetched["final"] == "pong"


def test_server_output_schema(server):
    url = f"{BASE}:{server.port}"
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    # Factory responds with plain prose -> schema mismatch flags it in the
    # result's structured_output field (None) rather than a crash.
    result = _post(url + "/runs", {"message": "answer", "output_schema": schema})
    assert result["structured_output"] is None


def test_server_bad_requests(server):
    url = f"{BASE}:{server.port}"
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(url + "/runs", {"message": "   "})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(url + "/runs/nope")
    assert err.value.code == 404


def test_server_token_enforced():
    srv = AgentServer(_factory, port=0, token="s3cret")
    srv.start_in_thread()
    try:
        url = f"{BASE}:{srv.port}"
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(url + "/runs", {"message": "ping"})
        assert err.value.code == 401
        ok = _post(url + "/runs", {"message": "ping"}, token="s3cret")
        assert ok["status"] == "completed"
    finally:
        srv.shutdown()


# --- EvalRunner ----------------------------------------------------------


def _eval_factory() -> Agent:
    return Agent(
        llm=MockLLMClient([text_response("42")]),
        permission_policy=PermissivePolicy(),
    )


def test_eval_runner_happy_path():
    runner = EvalRunner(_eval_factory)
    report = runner.run([EvalCase("answer?", expected="42")])
    assert report["total"] == 1 and report["passed"] == 1
    assert report["pass_rate"] == 1.0


def test_eval_runner_default_scorers():
    runner = EvalRunner(_eval_factory)
    report = runner.run([EvalCase("answer?", expected="nope")])
    assert report["passed"] == 0
    # contains_expected fails, no_tool_failures passes -> still not passed.
    scores = report["results"][0]["scores"]
    assert any(s["name"] == "no_tool_failures" for s in scores)


def test_eval_runner_tool_called_scorer():
    def factory() -> Agent:
        return Agent(
            llm=MockLLMClient(
                [
                    tool_response("c1", "echo", {"message": "x"}),
                    text_response("done"),
                ]
            ),
            tools=[EchoTool()],
            permission_policy=PermissivePolicy(),
        )

    runner = EvalRunner(
        factory,
        scorers=[tool_called("echo"), exact_match],
    )
    report = runner.run([EvalCase("echo x", expected="done")])
    assert report["passed"] == 1


def test_eval_runner_factory_crash_is_case_error():
    runner = EvalRunner(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    report = runner.run([EvalCase("x")])
    assert report["results"][0]["error"].startswith("agent_factory")


def test_eval_case_result_serialises():
    class NoScorers(Tool := object):
        pass

    runner = EvalRunner(_eval_factory, scorers=[exact_match])
    report = runner.run([EvalCase("a", expected="42", tags=["easy"])])
    r = report["results"][0]
    assert r["tags"] == ["easy"] and json.dumps(r) is not None


# --- Docker wrappers ------------------------------------------------------


def test_docker_exec_wrapper_builds_argv():
    argv = docker_exec_wrapper("dev-container", workdir="/app", user="agent")
    assert argv == ["docker", "exec", "-w", "/app", "-u", "agent", "-i", "dev-container"]


def test_docker_exec_wrapper_rejects_bad_name():
    with pytest.raises(SandboxWrapperError):
        docker_exec_wrapper("bad;name")


def test_docker_run_wrapper_defaults_offline():
    argv = docker_run_wrapper("python:3.12")
    assert argv[:4] == ["docker", "run", "--rm", "-i"]
    assert "--network" in argv and "none" in argv


def test_docker_run_wrapper_volumes_and_env():
    argv = docker_run_wrapper(
        "python:3.12",
        volumes=["/repo:/app"],
        env={"API_KEY": "x"},
        workdir="/app",
        network="host",
    )
    assert "-v" in argv and "/repo:/app" in argv
    assert "-e" in argv
    assert "--network" in argv and "host" in argv


def test_docker_run_wrapper_bad_volume():
    with pytest.raises(SandboxWrapperError):
        docker_run_wrapper("img", volumes=["just-a-path"])


def test_terminal_tool_command_wrapper_prefix_works():
    # env(1) is a harmless prefix that simply execs its argv as-is.
    tool = TerminalTool(command_wrapper=["env"])
    result = tool.execute(command="echo wrapped-then-run")
    assert not result.is_error and "wrapped-then-run" in result.output


# --- no_tool_failures scorer shape --------------------------------------


def test_no_tool_failures_scorer_with_error_tool():

    from tests._helpers import FailingTool

    def factory() -> Agent:
        return Agent(
            llm=MockLLMClient(
                [
                    tool_response("c1", "fail", {"message": "x"}),
                    text_response("ok"),
                ]
            ),
            tools=[FailingTool()],
            permission_policy=PermissivePolicy(),
        )

    runner = EvalRunner(factory, scorers=[no_tool_failures])
    report = runner.run([EvalCase("call fail")])
    assert report["results"][0]["passed"] is False
