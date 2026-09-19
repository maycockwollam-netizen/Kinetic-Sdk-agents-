"""Safety regressions for FileTool's conservative fuzzy replacement."""

from kinetic_sdk.files import FileTool
from kinetic_sdk.workspace.manager import LocalWorkspace


def test_anchored_block_refuses_a_far_larger_range(tmp_path):
    path = tmp_path / "module.py"
    original = "start\n" + "".join(f"filler_{number}\n" for number in range(104)) + "end\n"
    path.write_text(original)

    result = FileTool(LocalWorkspace(tmp_path)).execute(
        action="str_replace", path="module.py", old_str="start\nwrong\nend", new_str="changed"
    )

    assert result.is_error
    assert "more exact old_str" in str(result.error)
    assert path.read_text() == original


def test_anchored_block_replaces_a_similar_nearby_block_and_reports_lines(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("before\nstart\nactual = 1\nfinish\nafter\n")

    result = FileTool(LocalWorkspace(tmp_path)).execute(
        action="str_replace",
        path="module.py",
        old_str="start\nactual = 2\nfinish",
        new_str="start\nactual = 3\nfinish",
    )

    assert not result.is_error
    assert result.metadata["match_strategy"] == "anchored_block"
    assert "fuzzy=anchored_block" in result.output
    assert "lines 2-4" in result.output
    assert path.read_text() == "before\nstart\nactual = 3\nfinish\nafter\n"


def test_indentation_normalized_rewrites_new_text_using_tabs(tmp_path):
    path = tmp_path / "tabs.py"
    path.write_text("def run():\n\treturn value\n")

    result = FileTool(LocalWorkspace(tmp_path)).execute(
        action="str_replace",
        path="tabs.py",
        old_str="def run():\n    return value",
        new_str="def run():\n    return replacement",
    )

    assert not result.is_error
    assert path.read_text() == "def run():\n\treturn replacement\n"


def test_indentation_normalized_keeps_space_indentation(tmp_path):
    path = tmp_path / "spaces.py"
    path.write_text("def run():\n    return value\n")

    result = FileTool(LocalWorkspace(tmp_path)).execute(
        action="str_replace",
        path="spaces.py",
        old_str="def run():\n\treturn value",
        new_str="def run():\n\treturn replacement",
    )

    assert not result.is_error
    assert path.read_text() == "def run():\n    return replacement\n"


def test_file_tool_preserves_crlf_when_replacing_multiline_text(tmp_path):
    path = tmp_path / "windows.txt"
    path.write_bytes(b"one\r\ntwo\r\n")

    result = FileTool(LocalWorkspace(tmp_path)).execute(
        action="str_replace", path="windows.txt", old_str="one\r\ntwo", new_str="ONE\nTWO"
    )

    assert not result.is_error
    assert path.read_bytes() == b"ONE\r\nTWO\r\n"
