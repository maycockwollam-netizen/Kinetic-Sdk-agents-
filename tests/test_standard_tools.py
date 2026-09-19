"""Tests for the standard coding-agent tools: TerminalTool and FileTool.

Everything runs real code paths: real subprocesses for the terminal, real
files under ``tmp_path`` workspaces for the file editor.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.files import FileTool, GlobTool, GrepTool
from kinetic_sdk.terminal import TerminalTool
from kinetic_sdk.workspace.manager import Workspace


@pytest.fixture()
def workspace(tmp_path):
    return Workspace(tmp_path)


# --- TerminalTool ---------------------------------------------------------------


def test_terminal_runs_command_and_captures_output():
    tool = TerminalTool()
    result = tool.execute(command="echo hello && echo err >&2")
    assert not result.is_error
    assert "hello" in result.output
    assert "err" in result.output  # stderr is merged
    assert result.metadata["exit_code"] == 0


def test_terminal_nonzero_exit_is_error_result():
    result = TerminalTool().execute(command="echo partial; exit 3")
    assert result.is_error
    assert "code 3" in result.error
    assert "partial" in result.output  # partial output is preserved
    assert result.metadata["exit_code"] == 3


def test_terminal_timeout_kills_process_group():
    tool = TerminalTool(timeout=0.2)
    result = tool.execute(command="sleep 30")
    assert result.is_error
    assert "timed out" in result.error
    assert result.metadata["timed_out"] is True


def test_terminal_refuses_local_workspace_that_is_not_a_sandbox(workspace):
    tool = TerminalTool(workspace=workspace)
    result = tool.execute(command="pwd")
    assert result.is_error
    assert "not a sandbox" in str(result.error)


def test_terminal_output_truncation():
    tool = TerminalTool(max_output_chars=1_000)
    result = tool.execute(command="python3 -c \"print('x' * 5000)\"")
    assert not result.is_error
    assert result.metadata["truncated"] is True
    assert "cắt bớt" in result.output
    assert len(result.output) < 1_200


def test_terminal_empty_command_rejected():
    result = TerminalTool().execute(command="   ")
    assert result.is_error
    assert "non-empty" in result.error


def test_terminal_per_call_timeout_is_capped():
    tool = TerminalTool(timeout=0.5, max_timeout=1.0)
    result = tool.execute(command="sleep 5", timeout=300)  # capped to 1.0s
    assert result.is_error
    assert result.metadata["timed_out"] is True


def test_terminal_constructor_validation():
    with pytest.raises(ValueError):
        TerminalTool(timeout=0)
    with pytest.raises(ValueError):
        TerminalTool(timeout=10, max_timeout=5)
    with pytest.raises(ValueError):
        TerminalTool(max_output_chars=10)


def test_terminal_confirmation_patterns_match_dangerous_commands():
    """The ready-made patterns must catch the classics via AllowListPolicy."""
    from kinetic_sdk.security.policy import AllowListPolicy

    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": TerminalTool.REQUIRE_CONFIRMATION_PATTERNS},
    )
    safe = policy.check("terminal", {"command": "ls -la"})
    assert safe.allowed and not safe.requires_confirmation
    for dangerous in ("rm -rf /", "sudo apt install x", "chmod -R 777 /"):
        decision = policy.check("terminal", {"command": dangerous})
        assert decision.allowed and decision.requires_confirmation, dangerous


# --- FileTool -------------------------------------------------------------------


def test_file_view_with_line_numbers(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    tool = FileTool(workspace)
    result = tool.execute(action="view", path="a.txt")
    assert not result.is_error
    assert "1\tone" in result.output
    assert "3\tthree" in result.output


def test_file_view_range(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    tool = FileTool(workspace)
    result = tool.execute(action="view", path="a.txt", view_range=[3, 5])
    assert "3\tline3" in result.output
    assert "5\tline5" in result.output
    assert "line6" not in result.output


def test_file_view_truncation_explains_how_to_continue(workspace, tmp_path):
    (tmp_path / "long.txt").write_text("\n".join(f"line{i}" for i in range(1, 13)))
    result = FileTool(workspace, max_view_lines=10).execute(action="view", path="long.txt")
    assert result.metadata["truncated"] is True
    assert "còn 2 dòng nữa" in result.output
    assert "view_range=[11, 12]" in result.output


def test_file_view_cuts_overlong_physical_line(workspace, tmp_path):
    (tmp_path / "minified.txt").write_text("x" * 600)
    result = FileTool(workspace).execute(action="view", path="minified.txt")
    assert "...(cut)" in result.output
    assert len(result.output) < 550


def test_file_view_directory(workspace, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.txt").write_text("x")
    result = FileTool(workspace).execute(action="view", path=".")
    assert "sub/" in result.output
    assert "f.txt" in result.output


def test_file_create_and_refuse_overwrite(workspace, tmp_path):
    tool = FileTool(workspace)
    result = tool.execute(action="create", path="new.txt", file_text="hello")
    assert not result.is_error
    assert (tmp_path / "new.txt").read_text() == "hello"

    again = tool.execute(action="create", path="new.txt", file_text="other")
    assert again.is_error
    assert "already exists" in again.error


def test_file_create_in_nested_missing_dirs(workspace, tmp_path):
    tool = FileTool(workspace)
    result = tool.execute(action="create", path="deep/nest/f.txt", file_text="x")
    assert not result.is_error
    assert (tmp_path / "deep" / "nest" / "f.txt").read_text() == "x"


def test_file_str_replace_unique_match(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("foo bar foo")
    tool = FileTool(workspace)

    ambiguous = tool.execute(action="str_replace", path="a.txt", old_str="foo", new_str="x")
    assert ambiguous.is_error
    assert "2 times" in ambiguous.error

    ok = tool.execute(action="str_replace", path="a.txt", old_str="bar foo", new_str="baz")
    assert not ok.is_error
    assert (tmp_path / "a.txt").read_text() == "foo baz"

    missing = tool.execute(action="str_replace", path="a.txt", old_str="nope", new_str="x")
    assert missing.is_error and "not found" in missing.error


def test_file_insert(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("one\nthree\n")
    tool = FileTool(workspace)
    result = tool.execute(action="insert", path="a.txt", insert_line=1, new_str="two")
    assert not result.is_error
    assert (tmp_path / "a.txt").read_text() == "one\ntwo\nthree\n"


def test_file_insert_out_of_range(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("one\n")
    tool = FileTool(workspace)
    result = tool.execute(action="insert", path="a.txt", insert_line=99, new_str="x")
    assert result.is_error and "out of range" in result.error


def test_file_undo_edit(workspace, tmp_path):
    (tmp_path / "a.txt").write_text("original")
    tool = FileTool(workspace)
    tool.execute(action="str_replace", path="a.txt", old_str="original", new_str="changed")
    assert (tmp_path / "a.txt").read_text() == "changed"

    result = tool.execute(action="undo_edit", path="a.txt")
    assert not result.is_error
    assert (tmp_path / "a.txt").read_text() == "original"


def test_file_undo_of_create_deletes_file(workspace, tmp_path):
    tool = FileTool(workspace)
    tool.execute(action="create", path="temp.txt", file_text="x")
    assert (tmp_path / "temp.txt").exists()
    FileTool(workspace)  # a fresh tool has no undo memory
    result = tool.execute(action="undo_edit", path="temp.txt")
    assert not result.is_error
    assert not (tmp_path / "temp.txt").exists()


def test_file_undo_nothing_to_undo(workspace):
    result = FileTool(workspace).execute(action="undo_edit", path="never.txt")
    assert result.is_error and "nothing to undo" in result.error


def test_file_path_traversal_rejected(workspace):
    tool = FileTool(workspace)
    for bad in ("../escape.txt", "/etc/passwd"):
        result = tool.execute(action="view", path=bad)
        assert result.is_error, bad
        assert "path rejected" in result.error


def test_file_symlink_escape_rejected(workspace, tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    result = FileTool(workspace).execute(action="view", path="link.txt")
    assert result.is_error
    outside.unlink()


def test_file_unknown_action(workspace):
    result = FileTool(workspace).execute(action="delete_everything", path="x.txt")
    assert result.is_error and "unknown action" in result.error


def test_file_view_missing_file(workspace):
    result = FileTool(workspace).execute(action="view", path="ghost.txt")
    assert result.is_error and "no such file" in result.error


# --- GrepTool / GlobTool ------------------------------------------------------


def test_grep_finds_matching_lines_and_honors_max_results(workspace, tmp_path):
    (tmp_path / "a.py").write_text("first needle\nsecond needle\n")
    result = GrepTool(workspace).execute(pattern="needle", glob="*.py", max_results=1)
    assert not result.is_error
    assert result.output == "a.py:1:first needle"
    assert result.metadata["count"] == 1


def test_grep_fallback_works_without_ripgrep(workspace, tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("Alpha\nbeta\n")
    monkeypatch.setattr("kinetic_sdk.files.search.shutil.which", lambda _: None)
    result = GrepTool(workspace).execute(pattern="alpha", case_sensitive=False)
    assert not result.is_error
    assert result.output == "notes.txt:1:Alpha"


def test_grep_rejects_path_traversal(workspace):
    result = GrepTool(workspace).execute(pattern="secret", path="../")
    assert result.is_error
    assert "outside the workspace" in result.error


def test_grep_cuts_overlong_matching_line(workspace, tmp_path, monkeypatch):
    (tmp_path / "log.txt").write_text("match " + "x" * 600)
    monkeypatch.setattr("kinetic_sdk.files.search.shutil.which", lambda _: None)
    result = GrepTool(workspace).execute(pattern="match")
    assert "...(cut)" in result.output
    assert len(result.output) < 530


def test_glob_finds_python_files_and_excludes_symlink_escape(workspace, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret\n")
    (tmp_path / "escaped.py").symlink_to(outside)

    result = GlobTool(workspace).execute(pattern="**/*.py")
    assert not result.is_error
    assert result.output == "src/main.py"
    outside.unlink()
