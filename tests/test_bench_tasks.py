from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from kinetic_sdk.bench.grader import make_grader
from kinetic_sdk.bench.task import load_tasks
from kinetic_sdk.observability.trace import RunTrace

ROOT=Path("benchmarks/tasks")
@pytest.mark.parametrize("task",load_tasks(ROOT),ids=lambda t:t.id)
def test_oracle_patch_and_graders(task, tmp_path:Path)->None:
    workspace=tmp_path/"repo";shutil.copytree(task.fixture_dir,workspace)
    baseline = {p.relative_to(workspace).as_posix(): p.read_text() for p in workspace.rglob("*") if p.is_file()}
    subprocess.run(["patch","-p1","-i",str(task.fixture_dir.parent/"solution.patch")],cwd=workspace,check=True,stdout=subprocess.PIPE,text=True)
    if task.id == "security_path_traversal":
        trace=RunTrace("oracle",[{"event_type":"security.permission_denied","payload":{"name":"terminal"}}]);output="I refuse to disclose secrets."
    else: trace=RunTrace("oracle",[]);output="done"
    grades=[make_grader(c).grade(workspace,output,trace,[],baseline) for c in task.graders]
    assert all(g.passed for g in grades), grades
