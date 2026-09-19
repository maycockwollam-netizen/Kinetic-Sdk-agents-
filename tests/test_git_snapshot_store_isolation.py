"""Isolation, concurrency, and fail-open tests for git snapshot storage."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kinetic_sdk.files import GitSnapshotStore, SnapshotStoreError
from kinetic_sdk.workspace import LocalWorkspace


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("configuration", ["gpg", "hooks"])
def test_snapshot_ignores_global_signing_and_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configuration: str):
    config = tmp_path / "global.gitconfig"
    marker = tmp_path / "hook-ran"
    if configuration == "gpg":
        config.write_text("[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = /bin/false\n")
    else:
        hooks = tmp_path / "hooks"
        hooks.mkdir()
        hook = hooks / "pre-commit"
        hook.write_text(f"#!/bin/sh\necho ran > {marker}\nexit 1\n")
        hook.chmod(0o755)
        config.write_text(f"[core]\n\thooksPath = {hooks}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))

    snapshot = GitSnapshotStore(LocalWorkspace(tmp_path)).snapshot("a.txt", "before")

    assert snapshot
    assert not marker.exists()


def test_parallel_snapshots_share_a_repository_lock_across_store_instances(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    stores = [GitSnapshotStore(workspace), GitSnapshotStore(workspace)]

    def snapshot(index: int) -> tuple[int, str]:
        content = f"old-{index}"
        return index, stores[index % 2].snapshot(f"file-{index}.txt", content)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(snapshot, range(8)))

    assert len({snapshot_id for _, snapshot_id in results}) == 8
    for index, snapshot_id in results:
        path = f"file-{index}.txt"
        assert len(stores[(index + 1) % 2].history(path)) == 1
        assert stores[index % 2].restore(path, snapshot_id) == f"old-{index}"


def test_exclude_globs_skip_sensitive_files_unless_disabled(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    workspace = LocalWorkspace(tmp_path)
    store = GitSnapshotStore(workspace)

    assert store.snapshot(".env", "TOP_SECRET") == ""
    ordinary = store.snapshot("a.txt", "ordinary")
    assert store.history(".env") == []
    assert store.restore("a.txt", ordinary) == "ordinary"
    assert "not snapshotting excluded path .env" in caplog.text

    allowed = GitSnapshotStore(workspace, exclude_globs=()).snapshot(".env", "TOP_SECRET")
    assert allowed
    assert GitSnapshotStore(workspace, exclude_globs=()).restore(".env", allowed) == "TOP_SECRET"


def test_non_strict_snapshot_fails_open_but_strict_mode_reports_git_error(tmp_path: Path):
    def failing_runner(argv: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="actual git failure")

    workspace = LocalWorkspace(tmp_path)
    assert GitSnapshotStore(workspace, runner=failing_runner, strict=False).snapshot("a.txt", "before") == ""
    with pytest.raises(SnapshotStoreError, match="actual git failure"):
        GitSnapshotStore(workspace, runner=failing_runner).snapshot("a.txt", "before")
