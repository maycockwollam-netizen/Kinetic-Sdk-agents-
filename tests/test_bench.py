from __future__ import annotations

from pathlib import Path

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.bench.grader import (
    CommandGrader,
    DiffConstraintsGrader,
    NoSecretLeakGrader,
)
from kinetic_sdk.bench.runner import BenchRunner
from kinetic_sdk.bench.task import BenchTask, load_task
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response


def _task(root: Path) -> BenchTask:
    (root / "repo").mkdir()
    (root / "repo" / "app.py").write_text("def value():\n    return 1\n")
    (root / "hidden_tests").mkdir()
    (root / "hidden_tests" / "test_hidden.py").write_text("assert True\n")
    (root / "task.toml").write_text('''id = "demo"\ntitle = "Demo"\ncategory = "test"\ndifficulty = "easy"\nprompt = "Say done"\nfixture_dir = "repo"\nhidden_tests_dir = "hidden_tests"\nallowed_tools = []\npolicy_profile = "default"\n[budget]\nmax_llm_calls = 2\n[graders.tests]\ncommand = "python hidden_tests/test_hidden.py"\n''')
    return load_task(root)


def test_loader_rejects_unknown_and_reports_field(tmp_path: Path) -> None:
    task = _task(tmp_path)
    assert task.id == "demo"
    path = tmp_path / "task.toml"
    path.write_text(path.read_text() + "unexpected = 1\n")
    try:
        load_task(tmp_path)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected strict configuration error")


def test_runner_copies_hidden_tests_only_after_agent(tmp_path: Path) -> None:
    task = _task(tmp_path)
    seen: list[bool] = []

    def factory(task: BenchTask, workspace: Path) -> Agent:
        seen.append((workspace / "hidden_tests" / "test_hidden.py").exists())
        return Agent(llm=MockLLMClient([text_response("done")]), permission_policy=PermissivePolicy())

    record = BenchRunner(factory, runs=1, unsafe_local_exec=True).run([task]).records[0]
    assert seen == [False]
    assert record.status == "pass"


def test_graders_and_runner_classify_failure_and_infra(tmp_path: Path) -> None:
    task = _task(tmp_path)
    def agent_factory(task: BenchTask, workspace: Path) -> Agent:
        return Agent(llm=MockLLMClient([text_response("no")]), permission_policy=PermissivePolicy())
    result = BenchRunner(agent_factory, runs=1, unsafe_local_exec=True).run([task])
    assert result.records[0].status == "pass"  # tests, not answer text, grade this task
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("canary")
    assert not NoSecretLeakGrader("canary").grade(workspace, "canary", None, []).passed
    assert not CommandGrader("false").grade(workspace, "", None, []).passed
    baseline = {"a.txt": "old"}
    assert not DiffConstraintsGrader(allowed_files=["b.txt"]).grade(workspace, "", None, [], baseline).passed
