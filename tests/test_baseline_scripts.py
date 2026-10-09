from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_mock_baseline_and_compact_summary(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = subprocess.run(
        ["bash", "scripts/run_baseline.sh"],
        cwd=ROOT,
        env={**os.environ, "MODEL": "mock", "PROFILE": "mock", "RUNS": "1", "OUTPUT_DIR": str(output), "PYTHON": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    results = output / "results.json"
    assert len(json.loads(results.read_text())["results"]) == 11
    compact = output / "baseline.json"
    assert subprocess.run([sys.executable, "scripts/make_baseline.py", str(results), "--output", str(compact)], cwd=ROOT).returncode == 0
    data = json.loads(compact.read_text())
    assert data["success_rate"] == 1.0
    assert len(data["per_task"]) == 11
