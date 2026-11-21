"""Terminal UI for the Kinetic CLI.

``rich`` is optional: when installed the CLI renders with colours, a status
spinner and an evidence table; otherwise every helper degrades to plain
``print``/``input`` calls so the CLI stays usable on a zero-dependency core.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    _console: Console | None = Console()
    _error_console: Console | None = Console(stderr=True)
except ImportError:  # rich is optional
    _console = None
    _error_console = None


def print_system(message: str) -> None:
    if _console is not None:
        _console.print(f"[bold cyan]▸ {escape(message)}[/bold cyan]")
    else:
        print(f"▸ {message}")


def print_user(message: str) -> None:
    if _console is not None:
        _console.print(f"[bold green]you>[/bold green] {escape(message)}")
    else:
        print(f"you> {message}")


def print_assistant(message: str) -> None:
    if _console is not None:
        _console.print(message, markup=False)
    else:
        print(message)


def print_tool(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    if _console is not None:
        _console.print(f"[bold yellow]tool[/bold yellow] {escape(name)}{escape(suffix)}")
    else:
        print(f"[tool] {name}{suffix}")


def print_error(message: str) -> None:
    if _error_console is not None:
        _error_console.print(f"[bold red]error[/bold red] {escape(message)}")
    else:
        print(f"error: {message}", file=sys.stderr)


def print_ok(message: str) -> None:
    if _console is not None:
        _console.print(f"[bold green]{escape(message)}[/bold green]")
    else:
        print(message)


def stream_delta(delta: str) -> None:
    """Print a text-streaming chunk without a trailing newline."""
    if _console is not None:
        _console.print(delta, end="", markup=False)
    else:
        print(delta, end="", flush=True)


@contextmanager
def status(message: str) -> Iterator[None]:
    """Wrap a long-running section with a spinner (or a plain ``...`` line)."""
    if _console is not None:
        with _console.status(message, spinner="dots"):
            yield
    else:
        print(f"{message}...")
        yield


def print_evidence_panel(evidence: Any) -> None:
    """Render verification evidence; ``None`` renders as a plain notice."""
    if evidence is None:
        if _console is not None:
            _console.print("[dim]Evidence: none[/dim]")
        else:
            print("Evidence: none")
        return
    rows = [
        ("changed_files", evidence.changed_files),
        ("commands_run", evidence.commands_run),
        ("test_results", evidence.test_results),
        ("remaining_risks", evidence.remaining_risks),
        ("verified", evidence.is_verified()),
    ]
    if _console is not None:
        table = Table(title="Evidence", show_header=False)
        table.add_column(style="bold")
        table.add_column()
        for name, value in rows:
            table.add_row(name, str(value))
        _console.print(table)
    else:
        print("Evidence:")
        for name, value in rows:
            print(f"  {name}: {value}")


def print_banner() -> None:
    if _console is not None:
        _console.print(Panel("Kinetic interactive chat — /help for commands, /exit to quit.", style="bold cyan"))
    else:
        print("Kinetic interactive chat. Type /help for commands, /exit to quit.")