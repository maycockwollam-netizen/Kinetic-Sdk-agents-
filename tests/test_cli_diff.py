"""Tests for the CLI review-diff feature (kinetic diff / --diff)."""

from __future__ import annotations

import subprocess

from kinetic_sdk.cli.main import _workspace_diff, cmd_diff


def _git(tmp_path, *args: str) -> None:
    subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)


def _init_repo(tmp_path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "file.txt").write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")


def test_workspace_diff_shows_changes(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("one\nchanged\n", encoding="utf-8")

    diff = _workspace_diff(tmp_path)
    assert "file.txt" in diff
    assert "changed" in diff


def test_workspace_diff_empty_when_clean(tmp_path) -> None:
    _init_repo(tmp_path)
    assert _workspace_diff(tmp_path) == ""


def test_workspace_diff_empty_without_git(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    assert _workspace_diff(tmp_path) == ""


def test_cmd_diff_prints_changes(tmp_path, capsys) -> None:
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("one\nedited\n", encoding="utf-8")

    assert cmd_diff(tmp_path) == 0
    assert "file.txt" in capsys.readouterr().out


def test_cmd_diff_clean_notice(tmp_path, capsys) -> None:
    _init_repo(tmp_path)
    assert cmd_diff(tmp_path) == 0
    assert "No working-tree changes" in capsys.readouterr().out