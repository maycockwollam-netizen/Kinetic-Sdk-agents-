"""Terminal UI helpers must never crash, with or without ``rich``."""

from __future__ import annotations

from kinetic_sdk.cli import ui


def test_cli_ui_helpers_write_plain_output(capfd) -> None:
    ui.print_system("booted")
    ui.print_user("me")
    ui.print_assistant("text with [markup] that should stay literal")
    ui.print_ok("done")
    ui.print_error("boom")
    ui.stream_delta("delta")
    ui.stream_delta(" chunk")
    captured = capfd.readouterr()
    assert "booted" in captured.out
    assert "that should stay literal" in captured.out
    assert "done" in captured.out
    assert "boom" in captured.err
    assert "delta chunk" in captured.out


def test_cli_ui_evidence_panel_none(capfd) -> None:
    ui.print_evidence_panel(None)
    assert "Evidence: none" in capfd.readouterr().out


def test_cli_ui_status_and_tool_do_not_raise() -> None:
    with ui.status("Working"):
        ui.print_tool("grep", "pattern=foo")
    assert True