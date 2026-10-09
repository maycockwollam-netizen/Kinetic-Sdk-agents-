"""Deterministic helpers for validating filesystem benchmark tasks."""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    root: Path

    @property
    def fixture_dir(self) -> Path:
        return self.root / "fixture"

    @property
    def hidden_tests_dir(self) -> Path:
        return self.root / "hidden_tests"

    @property
    def metadata_path(self) -> Path:
        return self.root / "task.json"

    @property
    def patch_path(self) -> Path:
        return self.root / "solution.patch"


def discover_tasks(tasks_root: Path) -> list[BenchmarkTask]:
    return [BenchmarkTask(path.name, path) for path in sorted(tasks_root.iterdir()) if (path / "task.json").is_file()]


def prepare_workspace(task: BenchmarkTask, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(task.fixture_dir, destination)
    return destination


def apply_solution(task: BenchmarkTask, destination: Path) -> Path:
    workspace = prepare_workspace(task, destination)
    result = subprocess.run(
        ["patch", "-p1", "-i", str(task.patch_path.resolve())], cwd=workspace, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"cannot apply oracle for {task.name}: {result.stderr}")
    return workspace


def run_task_tests(task: BenchmarkTask, workspace: Path) -> subprocess.CompletedProcess[str]:
    if not workspace.exists():
        prepare_workspace(task, workspace)
    hidden_destination = workspace / "_grader_hidden_tests"
    if hidden_destination.exists():
        shutil.rmtree(hidden_destination)
    shutil.copytree(task.hidden_tests_dir, hidden_destination)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests", "_grader_hidden_tests"], cwd=workspace, text=True, capture_output=True, check=False
    )
