"""Tests for atomic multi-file unified-diff application."""

from __future__ import annotations

from kinetic_sdk.files import ApplyPatchTool
from kinetic_sdk.workspace.manager import Workspace


def test_apply_patch_updates_all_files_and_reports_paths(tmp_path):
    (tmp_path / "one.py").write_text("value = 1\n")
    (tmp_path / "two.py").write_text("from one import value\n")
    patch = """--- a/one.py
+++ b/one.py
@@ -1 +1 @@
-value = 1
+value = 2
--- a/two.py
+++ b/two.py
@@ -1 +1 @@
-from one import value
+from one import new_value
"""

    result = ApplyPatchTool(Workspace(tmp_path)).execute(patch=patch)

    assert not result.is_error
    assert (tmp_path / "one.py").read_text() == "value = 2\n"
    assert (tmp_path / "two.py").read_text() == "from one import new_value\n"
    assert result.metadata["files"] == ["one.py", "two.py"]


def test_apply_patch_dry_run_keeps_first_file_when_later_context_mismatches(tmp_path):
    (tmp_path / "one.txt").write_text("before\n")
    (tmp_path / "two.txt").write_text("actual\n")
    patch = """--- a/one.txt
+++ b/one.txt
@@ -1 +1 @@
-before
+after
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-expected
+after
"""

    result = ApplyPatchTool(Workspace(tmp_path)).execute(patch=patch)

    assert result.is_error
    assert "two.txt" in str(result.error)
    assert (tmp_path / "one.txt").read_text() == "before\n"
    assert (tmp_path / "two.txt").read_text() == "actual\n"


def test_apply_patch_rejects_path_traversal_without_touching_workspace(tmp_path):
    (tmp_path / "safe.txt").write_text("safe\n")
    patch = """--- a/safe.txt
+++ b/safe.txt
@@ -1 +1 @@
-safe
+changed
--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""

    result = ApplyPatchTool(Workspace(tmp_path)).execute(patch=patch)

    assert result.is_error
    assert "outside the workspace" in str(result.error)
    assert (tmp_path / "safe.txt").read_text() == "safe\n"


def test_apply_patch_rejects_empty_and_malformed_patches(tmp_path):
    tool = ApplyPatchTool(Workspace(tmp_path))

    assert tool.execute(patch="").is_error
    malformed = tool.execute(patch="--- a/file.txt\n+++ b/file.txt\n")
    assert malformed.is_error
    assert "malformed" in str(malformed.error)


def test_apply_patch_rolls_back_written_files_when_later_write_fails(tmp_path, monkeypatch):
    (tmp_path / "one.txt").write_text("one\n")
    (tmp_path / "two.txt").write_text("two\n")
    workspace = Workspace(tmp_path)
    original_write = workspace.write_text
    calls = 0

    def fail_second_write(path: str, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        original_write(path, content)

    monkeypatch.setattr(workspace, "write_text", fail_second_write)
    patch = """--- a/one.txt
+++ b/one.txt
@@ -1 +1 @@
-one
+ONE
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-two
+TWO
"""

    result = ApplyPatchTool(workspace).execute(patch=patch)

    assert result.is_error
    assert "rolled back" in str(result.error)
    assert (tmp_path / "one.txt").read_text() == "one\n"
    assert (tmp_path / "two.txt").read_text() == "two\n"
