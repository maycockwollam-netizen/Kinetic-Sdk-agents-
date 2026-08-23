"""``TerminalTool``: run shell commands from the agent loop.

The tool executes the command verbatim through ``bash -c`` (shell semantics —
pipes, redirects, globs all work). It deliberately does NOT judge which
commands are dangerous: that is the permission policy's job, same as every
other tool. For convenience, :attr:`REQUIRE_CONFIRMATION_PATTERNS` mirrors the
``GitTool`` pattern — a ready-made list of dangerous-command patterns to plug
into ``AllowListPolicy(require_confirmation_patterns=...)`` so destructive
commands (``rm -rf``, ``sudo``, ``mkfs``, ``dd``, writes to block devices, ...)
require explicit confirmation via the ``ON_PERMISSION_CHECK`` hooks.

Guardrails the tool itself owns: wall-clock timeout (the subprocess group is
killed on expiry), output truncation (head + tail, so a flooded build log
never blows up the context), and an optional :class:`Workspace` the command
runs in.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Any, ClassVar

from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.manager import Workspace

logger = logging.getLogger(__name__)


class TerminalTool(Tool):
    """Execute a shell command and return its combined output.

    Args:
        workspace: Optional :class:`Workspace`; the command runs with
            ``cwd`` set to the workspace root. Without one, commands run in
            the current working directory.
        timeout: Default wall-clock seconds per command (the model may lower
            it per call via the ``timeout`` parameter, never raise it above
            ``max_timeout``).
        max_timeout: Hard cap on any requested timeout.
        max_output_chars: Combined stdout+stderr kept per call, head + tail
            style. Larger outputs are cut in the middle with a marker.
    """

    name: ClassVar[str] = "terminal"
    description: ClassVar[str] = (
        "Run a shell command (bash semantics: pipes, redirects, globs work). "
        "Returns combined stdout/stderr, the exit code and duration. Use for "
        "building, testing, searching and inspecting the environment."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Optional wall-clock seconds for this call "
                "(capped by the tool's max_timeout).",
            },
        },
        "required": ["command"],
    }

    #: Ready-made confirmation patterns for ``AllowListPolicy``. Matched
    #: against the JSON-serialised tool input.
    REQUIRE_CONFIRMATION_PATTERNS: ClassVar[list[str]] = [
        r"rm\s+-[a-zA-Z]*[rf]",  # rm -rf / rm -fr / ...
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\b\s+.*of=/dev/",
        r">\s*/dev/sd[a-z]",
        r"\bshutdown\b|\breboot\b|\bpoweroff\b",
        r":\(\)\s*\{",  # fork bomb
        r"\bchmod\s+-R\b",
        r"\bchown\s+-R\b",
    ]

    DEFAULT_TIMEOUT: ClassVar[float] = 120.0
    MAX_TIMEOUT: ClassVar[float] = 600.0
    DEFAULT_MAX_OUTPUT_CHARS: ClassVar[int] = 30_000

    def __init__(
        self,
        workspace: Workspace | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_timeout: float = MAX_TIMEOUT,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        env: dict[str, str] | None = None,
    ) -> None:
        if timeout <= 0 or max_timeout <= 0:
            raise ValueError("timeout values must be positive")
        if timeout > max_timeout:
            raise ValueError("timeout must not exceed max_timeout")
        if max_output_chars < 200:
            raise ValueError("max_output_chars must be >= 200")
        self.workspace = workspace
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.max_output_chars = max_output_chars
        #: Environment for the subprocess. ``None`` inherits the process env
        #: (the subprocess default); a dict REPLACES it. The tool never reads
        #: ``os.environ`` itself — callers decide what the shell may see.
        self.env = env

    def execute(
        self, command: str, timeout: float | None = None, **_: Any
    ) -> ToolResult:
        if not isinstance(command, str) or not command.strip():
            return ToolResult(error="command must be a non-empty string")
        effective_timeout = (
            self.timeout if timeout is None else min(float(timeout), self.max_timeout)
        )
        if effective_timeout <= 0:
            return ToolResult(error="timeout must be positive")

        cwd = self.workspace.root_path if self.workspace is not None else None
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,  # own process group -> kill the whole tree
                env=self.env,
            )
        except OSError as exc:
            return ToolResult(error=f"failed to start shell: {exc}")

        try:
            output, _ = proc.communicate(timeout=effective_timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            output, _ = proc.communicate()

        duration = time.monotonic() - started
        output, truncated = self._truncate(output or "")
        metadata: dict[str, Any] = {
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 3),
            "timed_out": timed_out,
            "truncated": truncated,
        }
        if timed_out:
            return ToolResult(
                output=output,
                error=f"command timed out after {effective_timeout}s (process group killed)",
                metadata=metadata,
            )
        if proc.returncode != 0:
            return ToolResult(
                output=output,
                error=f"command exited with code {proc.returncode}",
                metadata=metadata,
            )
        return ToolResult(output=output, metadata=metadata)

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        head = self.max_output_chars * 2 // 3
        tail = self.max_output_chars - head
        omitted = len(text) - head - tail
        return (
            text[:head]
            + f"\n[... {omitted} ký tự ở giữa đã được cắt bớt ...]\n"
            + text[-tail:],
            True,
        )
