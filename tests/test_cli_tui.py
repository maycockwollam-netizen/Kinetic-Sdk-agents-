"""Tests for the opencode-style Textual TUI (textual is an optional dep)."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.cli.tui_app import SPLASH_NAME, ChatScreen, KineticTuiApp
from kinetic_sdk.testing import MockLLMClient, text_response


def _agent() -> Agent:
    return Agent(llm=MockLLMClient([text_response("hello from mock")]), tools=[])


@pytest.mark.asyncio
async def test_splash_screen_shows_opencode_name() -> None:
    import asyncio

    app = KineticTuiApp(_agent())
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        assert SPLASH_NAME == "opencode"
        title = app.screen.query_one("#splash-title")
        assert SPLASH_NAME in str(title.render())


@pytest.mark.asyncio
async def test_enter_starts_chat_screen() -> None:
    import asyncio

    app = KineticTuiApp(_agent())
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.press("enter")
        await pilot.pause()
        await asyncio.sleep(0.05)
        assert isinstance(app.screen, ChatScreen)
        assert app.screen.query_one("#prompt") is not None


@pytest.mark.asyncio
async def test_chat_runs_task_via_mock_agent() -> None:
    import asyncio

    app = KineticTuiApp(_agent())
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.05)
        await pilot.press("enter")
        await pilot.pause()
        await asyncio.sleep(0.05)
        app.screen.query_one("#prompt").value = "say hi"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await asyncio.sleep(0.25)
        await pilot.pause()
        view = app.screen.query_one("#chat-view")  # type: ignore[attr-defined]
        text = "\n".join(str(widget.render()) for widget in view.children)
        assert "say hi" in text