from __future__ import annotations

import json
from pathlib import Path

from kpi_sdk.bench.harness import apply_solution, discover_tasks, run_task_tests

ROOT = Path(__file__).parents[1]
TASKS = ROOT / "benchmarks" / "tasks"


def test_all_task_oracles_pass_and_unpatched_fixtures_fail(tmp_path: Path) -> None:
    tasks = discover_tasks(TASKS)
    assert len(tasks) == 11
    for task in tasks:
        before = run_task_tests(task, tmp_path / "before")
        assert before.returncode != 0, task.name
        workspace = apply_solution(task, tmp_path / "solved")
        after = run_task_tests(task, workspace)
        assert after.returncode == 0, f"{task.name}: {after.stdout}\n{after.stderr}"


def test_tasks_do_not_expose_hidden_tests_to_agents() -> None:
    for task in discover_tasks(TASKS):
        metadata = json.loads(task.metadata_path.read_text(encoding="utf-8"))
        allowed = metadata["agent_workspace_files"]
        assert not any("hidden_tests" in entry for entry in allowed)
        assert task.hidden_tests_dir.name == "hidden_tests"
        assert task.hidden_tests_dir.is_dir()


def test_apply_solution_with_relative_tasks_root(tmp_path: Path, monkeypatch) -> None:
    """Regression: relative patch_path must survive the cwd=workspace switch.

    ``discover_tasks(Path("benchmarks/tasks"))`` yields relative paths; the
    ``patch`` subprocess runs with ``cwd=workspace``, so an unresolved
    ``-i`` path made every oracle fail with "No such file".
    """
    import os

    monkeypatch.chdir(ROOT)
    assert not os.path.isabs("benchmarks/tasks")
    task = next(t for t in discover_tasks(Path("benchmarks/tasks")) if t.name == "bugfix_currency_rounding")
    workspace = apply_solution(task, tmp_path / "ws")
    assert (workspace / "src" / "money.py").is_file()


def test_classify_real_status_separates_agent_failure_from_infra() -> None:
    from kpi_sdk.bench.schema import classify_real_status

    assert classify_real_status(0) == "passed"
    assert classify_real_status(1) == "failed"
    for code in (2, 124, 137, 255, -9):
        assert classify_real_status(code) == "infra_error", code
