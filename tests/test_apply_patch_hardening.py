"""Hardening regressions for unified-diff application."""

from kinetic_sdk.files import ApplyPatchTool
from kinetic_sdk.workspace.manager import LocalWorkspace


def _apply(tmp_path, patch: str):
    return ApplyPatchTool(LocalWorkspace(tmp_path)).execute(patch=patch)


def test_sections_for_one_path_are_applied_in_order(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a\nb\nh\n")
    result = _apply(tmp_path, """--- a/f.txt
+++ b/f.txt
@@ -1 +1 @@
-a
+A
--- a/f.txt
+++ b/f.txt
@@ -3 +3 @@
-h
+H
""")
    assert not result.is_error
    assert path.read_text() == "A\nb\nH\n"
    assert result.metadata["files"] == ["f.txt"]


def test_patch_preserves_crlf_bytes(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    result = _apply(tmp_path, "--- a/f.txt\n+++ b/f.txt\n@@ -2 +2 @@\n-two\n+TWO\n")
    assert not result.is_error
    assert path.read_bytes() == b"one\r\nTWO\r\n"


def test_patch_fuzzes_unique_nearby_context_but_not_far_context(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("extra\na\nb\nc\n")
    near = _apply(tmp_path, "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n")
    assert not near.is_error
    assert path.read_text() == "extra\na\nB\nc\n"

    path.write_text("".join(f"line {number}\n" for number in range(40)) + "a\nb\nc\n")
    far = _apply(tmp_path, "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n")
    assert far.is_error
    assert "within 20 lines" in str(far.error)


def test_patch_rejects_ambiguous_fuzzy_context(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("x\na\nb\nc\ny\na\nb\nc\n")
    result = _apply(tmp_path, "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n")
    assert result.is_error
    assert "multiple" in str(result.error)


def test_patch_accepts_blank_context_and_no_final_newline_marker(tmp_path):
    blank = tmp_path / "blank.txt"
    blank.write_text("before\n\nafter\n")
    result = _apply(tmp_path, "--- a/blank.txt\n+++ b/blank.txt\n@@ -1,3 +1,3 @@\n before\n \n-after\n+AFTER\n")
    assert not result.is_error
    assert blank.read_text() == "before\n\nAFTER\n"

    final = tmp_path / "final.txt"
    final.write_bytes(b"old")
    result = _apply(tmp_path, "--- a/final.txt\n+++ b/final.txt\n@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n")
    assert not result.is_error
    assert final.read_bytes() == b"new"
