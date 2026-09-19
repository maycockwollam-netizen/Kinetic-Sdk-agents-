"""Tests for the private, opt-in git snapshot backend."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kinetic_sdk.files import (
    FileHistoryTool,
    FileTool,
    GitSnapshotStore,
    GlobTool,
    GrepTool,
    SnapshotStoreError,
)
from kinetic_sdk.workspace import LocalWorkspace


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_snapshot_history_restores_each_durable_version(tmp_path: Path):
    workspace = LocalWorkspace(tmp_path)
    store = GitSnapshotStore(workspace)
    snapshots = [store.snapshot("a.txt", content) for content in ("one", "two", "three")]

    history = store.history("a.txt")
    assert [entry.snapshot_id for entry in history] == list(reversed(snapshots))
    assert [store.restore("a.txt", snapshot) for snapshot in snapshots] == ["one", "two", "three"]


def test_sidecar_is_private_ignored_and_in_a_different_git_repository(tmp_path: Path):
    git(tmp_path, "init", "--quiet")
    user_ignore = tmp_path / ".gitignore"
    user_ignore.write_text("*.pyc\n")
    before_ignore = user_ignore.read_bytes()
    workspace = LocalWorkspace(tmp_path)
    store = GitSnapshotStore(workspace)
    store.snapshot("a.txt", "before")

    assert (tmp_path / ".kinetic" / "snapshots" / ".git").is_dir()
    assert (tmp_path / ".kinetic" / "snapshots" / ".git") != tmp_path / ".git"
    assert (tmp_path / ".kinetic" / ".gitignore").read_text() == "*\n"
    assert user_ignore.read_bytes() == before_ignore
    assert ".kinetic" not in GlobTool(workspace).execute(pattern="**/*").output
    assert "before" not in GrepTool(workspace).execute(pattern="before").output
    assert ".kinetic" not in FileTool(workspace).execute(action="view", path=".").output
    assert ".kinetic" not in git(tmp_path, "status", "--short").stdout


def test_sidecar_does_not_create_or_modify_user_gitignore(tmp_path: Path):
    git(tmp_path, "init", "--quiet")
    store = GitSnapshotStore(LocalWorkspace(tmp_path))

    store.snapshot("a.txt", "before")
    store.snapshot("a.txt", "after")

    assert not (tmp_path / ".gitignore").exists()
    assert ".kinetic" not in git(tmp_path, "status", "--porcelain").stdout


def test_file_tool_snapshots_pre_edit_content_and_history_tool_restores(tmp_path: Path):
    (tmp_path / "a.txt").write_text("one")
    workspace = LocalWorkspace(tmp_path)
    store = GitSnapshotStore(workspace)
    tool = FileTool(workspace, snapshot_store=store)

    assert not tool.execute(action="str_replace", path="a.txt", old_str="one", new_str="two").is_error
    assert not tool.execute(action="str_replace", path="a.txt", old_str="two", new_str="three").is_error
    entries = store.history("a.txt")
    assert [store.restore("a.txt", entry.snapshot_id) for entry in entries] == ["two", "one"]

    restored = FileHistoryTool(workspace, store).execute(
        action="restore", path="a.txt", snapshot_id=entries[-1].snapshot_id
    )
    assert not restored.is_error
    assert (tmp_path / "a.txt").read_text() == "one"


def test_snapshot_history_is_strictly_opt_in(tmp_path: Path):
    (tmp_path / "a.txt").write_text("one")
    workspace = LocalWorkspace(tmp_path)
    result = FileTool(workspace).execute(action="str_replace", path="a.txt", old_str="one", new_str="two")
    assert not result.is_error
    assert not (tmp_path / ".kinetic").exists()
    assert FileHistoryTool(workspace).execute(action="list", path="a.txt").error == "snapshot history not enabled for this workspace"


def test_snapshot_rejects_path_traversal_and_invalid_snapshot_ids(tmp_path: Path):
    store = GitSnapshotStore(LocalWorkspace(tmp_path))
    snapshot = store.snapshot("a.txt", "before")

    with pytest.raises(ValueError):
        store.snapshot("../secret", "no")
    with pytest.raises(ValueError):
        store.snapshot(".kinetic/private", "no")
    with pytest.raises(ValueError):
        store.restore("a.txt", "not-a-commit")
    with pytest.raises(SnapshotStoreError):
        store.restore("missing.txt", snapshot)
