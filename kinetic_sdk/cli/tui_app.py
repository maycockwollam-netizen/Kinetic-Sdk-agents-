"""Textual TUI for the Kinetic CLI.

An opencode-style fullscreen interface: a splash screen, then a chat screen
with a scrolling conversation, a prompt input docked at the bottom and a
footer with key bindings. The agent runs on a background thread so the UI
never blocks; streaming deltas are applied through ``call_from_thread``.

``textual`` is optional (pyproject extra ``ui``); the CLI degrades to the
plain rich/print interface when it is missing.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from kinetic_sdk.agent.agent import Agent

SPLASH_NAME = "gow"


class ChatView(VerticalScroll):
    """Scrollable conversation: one ``Static`` per message/streamed reply."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current: Static | None = None
        self._buffer = ""

    def add_user(self, text: str) -> None:
        self._append_prefix(Text("you> ", style="bold green"), text, klass="msg user")

    def add_system(self, text: str) -> None:
        self._append_prefix(Text("▸ ", style="bold cyan"), text, klass="msg system")

    def add_error(self, text: str) -> None:
        self._append_prefix(Text("error> ", style="bold red"), text, klass="msg error")

    def open_assistant(self) -> None:
        self._current = Static("", id="stream", classes="msg assistant")
        self.mount(self._current)
        self._buffer = ""
        self.scroll_end()

    def stream(self, delta: str) -> None:
        if self._current is None:
            self.open_assistant()
        self._buffer += delta
        assert self._current is not None
        self._current.update(self._buffer)
        self.scroll_end()

    @property
    def has_output(self) -> bool:
        return bool(self._buffer)

    def _append_prefix(self, prefix: Text, text: str, *, klass: str) -> None:
        message = Text().append_text(prefix).append(text)
        self.mount(Static(message, classes=klass, markup=False))
        self.scroll_end()


class SplashScreen(Screen):
    """Startup screen showing the product title, like opencode's splash."""

    BINDINGS = [("enter", "go_chat", "Start"), ("escape", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Static(SPLASH_NAME, id="splash-title", classes="splash-title")
        yield Static("AI coding agent — press Enter to start", id="splash-hint", classes="splash-hint")

    def on_mount(self) -> None:
        self.set_timer(1.4, self.action_go_chat)

    def action_go_chat(self) -> None:
        cast(KineticTuiApp, self.app).switch_to_chat()


class ChatScreen(Screen):
    BINDINGS = [
        ("ctrl+q", "app.quit", "Quit"),
        ("ctrl+r", "reset", "Reset"),
        ("ctrl+l", "clear", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # opencode-style frame: sidebar (sessions) + chat pane + composer.
        yield Horizontal(
            VerticalScroll(
                Static("Sessions", classes="sidebar-title"),
                Static("▸ default session", classes="sidebar-item"),
                id="sidebar-view",
                classes="sidebar-view",
            ),
            Vertical(
                ChatView(id="chat-view"),
                Static("", id="composer-model", classes="composer-model"),
                Input(placeholder="Ask anything… (/help, /reset, /exit)", id="prompt"),
                id="chat-pane",
                classes="chat-pane",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        app = cast(KineticTuiApp, self.app)
        model = str(getattr(app.agent.llm, "model", "") or "unknown")
        self.query_one("#composer-model", Static).update(f"model: {model}")
        self.query_one("#prompt", Input).focus()
        view = self.query_one("#chat-view", ChatView)
        view.add_system("Connected. /help for commands, /exit to quit.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text in ("/exit", "/quit"):
            self.app.exit()
            return
        if text in ("/help", "/h"):
            self.query_one("#chat-view", ChatView).add_system(
                "Commands: /exit, /reset, /clear, /help"
            )
            return
        if text in ("/reset", "/clear"):
            self.query_one("#chat-view", ChatView).remove_children(".msg")
            self.query_one("#chat-view", ChatView).add_system(
                "Conversation cleared. /help for commands, /exit to quit."
            )
            return
        asyncio.create_task(_run_task_async(self, text))


async def _run_task_async(screen: ChatScreen, task: str) -> Any:
    """Run the agent off the UI thread and stream deltas back in."""
    view = screen.query_one("#chat-view", ChatView)
    app = cast(KineticTuiApp, screen.app)
    agent = app.agent
    view.add_user(task)
    view.open_assistant()

    def _on_delta(event: Any) -> None:
        delta = str(event.payload.get("delta", ""))
        app.call_from_thread(view.stream, delta)

    agent.event_bus.subscribe("agent.text_delta", _on_delta)
    try:
        await asyncio.to_thread(agent.run, task, stream=True)
    except Exception as exc:  # noqa: BLE001 - surface failures to the UI
        app.call_from_thread(view.add_error, str(exc))
    finally:
        agent.event_bus.unsubscribe("agent.text_delta", _on_delta)
        if not view.has_output:
            app.call_from_thread(view.add_error, "no output produced")


class KineticTuiApp(App[None]):
    TITLE = SPLASH_NAME
    SUB_TITLE = "AI coding agent"

    CSS = """
    #splash-title {
        content-align: center middle;
        width: 100%;
        height: 1fr;
        text-style: bold;
        color: $accent;
    }
    #splash-hint {
        content-align: center middle;
        width: 100%;
        height: auto;
        color: $text-muted;
    }
    .sidebar-view {
        width: 24;
        height: 1fr;
        border-right: heavy;
        background: $surface;
    }
    .sidebar-title {
        width: 100%;
        padding: 1 1 0 1;
        text-style: bold;
        color: $text-muted;
    }
    .sidebar-item {
        width: 100%;
        padding: 0 1;
    }
    #chat-pane {
        height: 1fr;
        width: 1fr;
    }
    #chat-view {
        height: 1fr;
        width: 1fr;
        border: round $primary;
        margin: 1 1 0 1;
        padding: 0 1;
    }
    .msg { width: 100%; padding-top: 1; }
    .msg.user { color: $text; text-style: bold; }
    .msg.system { color: $text-muted; }
    .msg.error { color: $error; }
    #composer-model {
        width: 100%;
        padding: 0 1 0 2;
        color: $text-muted;
    }
    #prompt {
        border: round $accent;
        margin: 0 1 1 1;
    }
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent

    def on_mount(self) -> None:
        self.push_screen(SplashScreen())

    def switch_to_chat(self) -> None:
        self.switch_screen(ChatScreen())