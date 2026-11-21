#!/usr/bin/env python3
"""Run benchmark tasks with the deterministic mock or an external real runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from kpi_sdk.bench.harness import apply_solution, discover_tasks, run_task_tests
from kpi_sdk.bench.schema import BenchmarkReport, TaskResult, classify_real_status


def git_sha() -> str:
    """Current HEAD, or ``unknown`` for a non-git checkout (e.g. an sdist)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    model = os.environ.get("MODEL", "mock")
    profile = os.environ.get("PROFILE", "mock")
    runs = int(os.environ.get("RUNS", "1"))
    if runs < 1:
        raise SystemExit("RUNS must be >= 1")
    default = ROOT / "benchmarks" / "baselines" / f"{model.replace('/', '_')}_{date.today().isoformat()}"
    output_dir = Path(os.environ.get("OUTPUT_DIR", default)); output_dir.mkdir(parents=True, exist_ok=True)
    report = BenchmarkReport(
        model=model,
        profile=profile,
        runs=runs,
        git_sha=git_sha(),
        date=date.today().isoformat(),
    )
    for task in discover_tasks(ROOT / "benchmarks" / "tasks"):
        for run in range(1, runs + 1):
            started = time.perf_counter()
            if profile == "mock":
                workspace = apply_solution(task, output_dir / "workspaces" / f"{task.name}-{run}")
                check = run_task_tests(task, workspace)
                status = "passed" if check.returncode == 0 else "failed"
                report.results.append(TaskResult(task.name, run, status, round(time.perf_counter() - started, 3), 0.0, {"secret_leaks": 0}, check.stdout + check.stderr))
            else:
                runner = os.environ.get("BENCH_REAL_RUNNER")
                if not os.environ.get("API_KEY"):
                    raise SystemExit("API_KEY is required unless PROFILE=mock")
                if not runner:
                    raise SystemExit("BENCH_REAL_RUNNER is required for a real profile")
                completed = subprocess.run([runner, str(task.root)], text=True, capture_output=True, check=False)
                status = classify_real_status(completed.returncode)
                report.results.append(TaskResult(task.name, run, status, round(time.perf_counter() - started, 3), None, {"secret_leaks": 0}, completed.stdout + completed.stderr))
    (output_dir / "results.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n")
    passed = sum(row.status == "passed" for row in report.results)
    (output_dir / "report.md").write_text(f"# Baseline {model}\n\n- Profile: `{profile}`\n- Passed: {passed}/{len(report.results)}\n- Success rate: {report.success_rate:.2f}\n- Results: `results.json`\n")
    print(output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
