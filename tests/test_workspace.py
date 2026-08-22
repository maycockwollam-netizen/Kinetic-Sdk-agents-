"""Tests for the workspace package: path-traversal safety and file listing.

All tests run inside pytest's tmp_path — no real system files are touched.
"""

from __future__ import annotations

import os

import pytest

from kinetic_sdk.workspace import PathTraversalError, Workspace


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# demo")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hi')")
    (root / "src" / "util").mkdir()
    (root / "src" / "util" / "helper.py").write_text("def f(): ...")
    (root / "notes.txt").write_text("hello")
    return Workspace(root)


# --- Construction -------------------------------------------------------------


def test_root_path_is_canonical(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    ws = Workspace(str(root) + os.sep + ".")
    assert ws.root_path == os.path.realpath(root)


def test_missing_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        Workspace(tmp_path / "does-not-exist")


def test_file_as_root_is_rejected(tmp_path):
    f = tmp_path / "a-file"
    f.write_text("x")
    with pytest.raises(ValueError, match="not an existing directory"):
        Workspace(f)


# --- resolve() ------------------------------------------------------------------


def test_resolve_valid_relative_path(workspace):
    resolved = workspace.resolve("src/main.py")
    assert resolved == os.path.join(workspace.root_path, "src", "main.py")


def test_resolve_nested_and_dot_segments_inside_root(workspace):
    assert workspace.resolve("src/util/../main.py") == os.path.join(
        workspace.root_path, "src", "main.py"
    )
    assert workspace.resolve(".") == workspace.root_path
    assert workspace.resolve("src/./util/../../notes.txt") == os.path.join(
        workspace.root_path, "notes.txt"
    )


def test_resolve_absolute_path_inside_root_is_allowed(workspace):
    inside = os.path.join(workspace.root_path, "src", "main.py")
    assert workspace.resolve(inside) == inside


@pytest.mark.parametrize(
    "evil",
    [
        "../../../etc/passwd",
        "..",
        "../..",
        "src/../../outside.txt",
        "src/../../../../etc/shadow",
    ],
)
def test_resolve_rejects_parent_traversal(workspace, evil):
    with pytest.raises(PathTraversalError, match="outside the workspace root"):
        workspace.resolve(evil)


def test_resolve_rejects_absolute_path_outside_root(workspace):
    with pytest.raises(PathTraversalError):
        workspace.resolve("/etc/passwd")


def test_resolve_rejects_symlink_escaping_root(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    os.symlink(outside, os.path.join(workspace.root_path, "link"))

    with pytest.raises(PathTraversalError):
        workspace.resolve("link/secret.txt")


def test_resolve_allows_symlink_staying_inside_root(workspace):
    os.symlink(
        os.path.join(workspace.root_path, "src"),
        os.path.join(workspace.root_path, "src-link"),
    )
    resolved = workspace.resolve("src-link/main.py")
    assert resolved == os.path.join(workspace.root_path, "src", "main.py")


def test_traversal_error_names_the_offending_path(workspace):
    with pytest.raises(PathTraversalError, match=r"\.\./\.\./etc/passwd"):
        workspace.resolve("../../etc/passwd")


# --- list_files() -----------------------------------------------------------------


def test_list_files_returns_all_files_sorted(workspace):
    assert workspace.list_files() == [
        "README.md",
        "notes.txt",
        "src/main.py",
        "src/util/helper.py",
    ]


def test_list_files_with_pattern(workspace):
    assert workspace.list_files("*.py") == ["src/main.py", "src/util/helper.py"]
    assert workspace.list_files("src/util/*") == ["src/util/helper.py"]
    assert workspace.list_files("*.does-not-exist") == []


def test_list_files_excludes_directories(workspace):
    os.mkdir(os.path.join(workspace.root_path, "empty-dir"))
    assert "empty-dir" not in workspace.list_files()
