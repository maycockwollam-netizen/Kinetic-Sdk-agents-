"""Regression tests for the offline benchmark harness and quality gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "bench_compare.py"
RUNNER = ROOT / "scripts" / "bench_run.py"
BASELINE = ROOT / "benchmarks" / "baselines" / "baseline.mock.json"
CASSETTES = ROOT / "benchmarks" / "cassettes"
GATE = ROOT / "benchmarks" / "gate.toml"


def _write_results(path: Path, tasks: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "tasks": tasks}), encoding="utf-8")


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    task: dict[str, object] = {
        "id": task_id,
        "groups": ["general"],
        "success": True,
        "cost_usd": 1.0,
        "secret_leaks": 0,
        "infra_error": False,
    }
    task.update(overrides)
    return task


def _compare(baseline: Path, results: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMPARE), str(baseline), str(results), "--gate", str(GATE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_mock_smoke_is_offline_and_deterministic(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    command = [sys.executable, str(RUNNER), "--mock-script", str(CASSETTES)]
    assert subprocess.run([*command, "--output", str(first)], cwd=ROOT, check=False).returncode == 0
    assert subprocess.run([*command, "--output", str(second)], cwd=ROOT, check=False).returncode == 0

    assert json.loads(first.read_text(encoding="utf-8")) == json.loads(second.read_text(encoding="utf-8"))


def test_gate_accepts_equal_or_better_results(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(BASELINE.read_text(encoding="utf-8"), encoding="utf-8")

    completed = _compare(BASELINE, result)

    assert completed.returncode == 0, completed.stdout
    assert "# Benchmark quality gate: PASS" in completed.stdout


def test_gate_rejects_security_failure_secret_leak_cost_and_infra_error(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_results(baseline, [_task("general"), _task("security", groups=["security", "red-team"])])
    _write_results(
        candidate,
        [
            _task("general", success=False, cost_usd=1.5, infra_error=True),
            _task("security", groups=["security", "red-team"], success=False, secret_leaks=1),
        ],
    )

    completed = _compare(baseline, candidate)

    assert completed.returncode == 1
    assert "security/red-team task failed: security" in completed.stdout
    assert "secret leak count is 1" in completed.stdout
    assert "cost per successful task increased" in completed.stdout
    assert "infra errors increased" in completed.stdout


def test_gate_tolerates_small_success_noise_inside_baseline_interval(tmp_path):
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    _write_results(baseline, [_task(str(index)) for index in range(20)])
    _write_results(candidate, [_task(str(index), success=index != 0) for index in range(20)])

    completed = _compare(baseline, candidate)

    assert completed.returncode == 0, completed.stdout
    assert "within the baseline confidence interval" in completed.stdout
