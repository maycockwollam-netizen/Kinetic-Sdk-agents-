"""Tests for the CLI `doctor` health command."""

from __future__ import annotations

from kinetic_sdk.cli.main import cmd_doctor


def test_cmd_doctor_reports_environment(capsys, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENHANDS_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert cmd_doctor() == 0
    out = capsys.readouterr().out
    assert "Python" in out
    assert "kinetic-agent-sdk" in out
    assert "Rust extension" in out
    assert "LLM API key" in out
    assert "OPENAI_API_KEY" in out