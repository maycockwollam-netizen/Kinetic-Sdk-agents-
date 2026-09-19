"""Regression coverage for bounded, workspace-safe search tools."""

from __future__ import annotations

import shutil
import time

import pytest

from kinetic_sdk.files import GlobTool, GrepTool
from kinetic_sdk.workspace import Workspace


@pytest.fixture
def search_workspace(tmp_path):
    return Workspace(tmp_path)


def _sample_tree(tmp_path):
    (tmp_path / "visible.txt").write_text("needle\n")
    (tmp_path / ".hidden.txt").write_text("needle\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n")
    (tmp_path / "binary.bin").write_bytes(b"needle\0not text")


def _grep_with_engine(workspace, monkeypatch, engine):
    monkeypatch.setattr("kinetic_sdk.files.search.shutil.which", lambda _: "rg" if engine == "rg" else None)
    return GrepTool(workspace).execute(pattern="needle")


def test_python_regex_redos_times_out_without_hanging(search_workspace, tmp_path, monkeypatch):
    (tmp_path / "slow.txt").write_text("a" * 26)
    monkeypatch.setattr("kinetic_sdk.files.search.shutil.which", lambda _: None)
    monkeypatch.setattr(GrepTool, "SEARCH_TIMEOUT_SECONDS", 1.0)

    started = time.monotonic()
    result = GrepTool(search_workspace).execute(pattern="(a+)+b")
    elapsed = time.monotonic() - started

    assert result.is_error
    assert "regex timed out" in result.error
    assert elapsed < 30


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_rg_and_python_skip_hidden_and_binary_files_equally(search_workspace, tmp_path, monkeypatch):
    _sample_tree(tmp_path)

    rg_result = _grep_with_engine(search_workspace, monkeypatch, "rg")
    python_result = _grep_with_engine(search_workspace, monkeypatch, "python")

    assert rg_result.output == python_result.output == "visible.txt:1:needle"
    assert rg_result.metadata["engine"] == "rg"
    assert python_result.metadata["engine"] == "python"


@pytest.mark.parametrize("engine", ["rg", "python"])
def test_grep_never_leaks_symlink_outside_workspace(search_workspace, tmp_path, monkeypatch, engine):
    if engine == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    outside = tmp_path.parent / "search-outside-secret.txt"
    outside.write_text("needle\n")
    (tmp_path / "escape.txt").symlink_to(outside)
    try:
        result = _grep_with_engine(search_workspace, monkeypatch, engine)
        assert not result.is_error
        assert "search-outside-secret" not in result.output
    finally:
        outside.unlink()


def test_grep_rejects_outside_workspace_path(search_workspace):
    result = GrepTool(search_workspace).execute(pattern="needle", path="../")
    assert result.is_error
    assert "outside the workspace" in result.error


@pytest.mark.parametrize("pattern", ["/tmp/*.py", "src/../*.py"])
def test_glob_rejects_absolute_and_parent_patterns_early(search_workspace, pattern):
    result = GlobTool(search_workspace).execute(pattern=pattern)
    assert result.is_error
    assert "pattern must" in result.error


def test_grep_limit_is_total_across_files(search_workspace, tmp_path, monkeypatch):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("needle\n")
    monkeypatch.setattr("kinetic_sdk.files.search.shutil.which", lambda _: None)

    result = GrepTool(search_workspace).execute(pattern="needle", max_results=2)

    assert result.output.splitlines() == ["a.txt:1:needle", "b.txt:1:needle"]
    assert result.metadata["count"] == 2


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_rg_limit_is_total_across_files(search_workspace, tmp_path):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("needle\n")

    result = GrepTool(search_workspace).execute(pattern="needle", max_results=2)

    assert len(result.output.splitlines()) == 2
    assert set(result.output.splitlines()).issubset(
        {"a.txt:1:needle", "b.txt:1:needle", "c.txt:1:needle"}
    )
    assert result.metadata["count"] == 2
