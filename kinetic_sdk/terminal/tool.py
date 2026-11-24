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
from typing import Any, Callable, ClassVar

from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

logger = logging.getLogger(__name__)


class TerminalTool(Tool):
    """Execute a shell command and return its combined output.

    Args:
        workspace: Optional :class:`WorkspaceBase`; the command runs through
            that backend. Without one, commands run on the local host in the
            current working directory.
        timeout: Default wall-clock seconds per command (the model may lower
            it per call via the ``timeout`` parameter, never raise it above
            ``max_timeout``).
        max_timeout: Hard cap on any requested timeout.
        max_output_chars: Combined stdout+stderr kept per call, head + tail
            style. Larger outputs are cut in the middle with a marker.
    """

    name: str = "terminal"
    description: str = (
        "Run a shell command (bash semantics: pipes, redirects, globs work). "
        "Returns combined stdout/stderr, the exit code and duration. Use for "
        "building, testing, searching and inspecting the environment."
    )
    parameters: dict[str, Any] = {
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
            "cwd": {
                "type": "string",
                "description": "Optional directory to run in (resolved through "
                "the workspace when one is configured).",
            },
            "env": {
                "type": "object",
                "description": "Optional extra environment variables for this "
                "call (merged over the tool-level env).",
            },
            "max_output_chars": {
                "type": "integer",
                "description": "Optional per-call output cap (>= 200).",
            },
            "truncation": {
                "type": "string",
                "enum": ["both", "head", "tail"],
                "description": "Where to keep output when it exceeds the cap: "
                "'both' (head+tail, default), 'head', or 'tail' (keep the end — "
                "good for build logs where the error is last).",
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

    #: Stateless per call: each execution spawns its own subprocess and the
    #: remaining state is immutable configuration, so one instance can be
    #: shared across concurrent runs.
    thread_safe = True

    def __init__(
        self,
        workspace: WorkspaceBase | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_timeout: float = MAX_TIMEOUT,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        env: dict[str, str] | None = None,
        command_wrapper: list[str] | None = None,
        truncation: str = "both",
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if timeout <= 0 or max_timeout <= 0:
            raise ValueError("timeout values must be positive")
        if timeout > max_timeout:
            raise ValueError("timeout must not exceed max_timeout")
        if max_output_chars < 200:
            raise ValueError("max_output_chars must be >= 200")
        if truncation not in ("both", "head", "tail"):
            raise ValueError("truncation must be 'both', 'head' or 'tail'")
        self.workspace = workspace
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.max_output_chars = max_output_chars
        self.truncation = truncation
        #: Optional callback invoked with each produced output line as it
        #: arrives (library/UI streaming; the agent loop is wired later).
        self.progress_callback = progress_callback
        #: Optional argv prefix placed before ``bash -c <command>`` — a
        #: sandbox adapter (e.g. ``docker exec -i <container>`` from
        #: ``terminal/docker.py``) turns itself into exactly this prefix,
        #: so the tool executes ``<wrapper> bash -c <command>`` while its
        #: own timeout/process-group kill keeps working on the outer layer.
        self.command_wrapper = list(command_wrapper or [])
        #: Environment for the subprocess. ``None`` inherits the process env
        #: (the subprocess default); a dict REPLACES it. The tool never reads
        #: ``os.environ`` itself — callers decide what the shell may see.
        self.env = env

    def execute(  # type: ignore[override]
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_output_chars: int | None = None,
        truncation: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> ToolResult:
        # Named-parameter signature by design: the agent loop always invokes
        # tools via execute(**model_arguments) after schema validation, so
        # narrowing the base **params contract is safe here.
        if not isinstance(command, str) or not command.strip():
            return ToolResult(error="command must be a non-empty string")
        effective_timeout = (
            self.timeout if timeout is None else min(float(timeout), self.max_timeout)
        )
        if effective_timeout <= 0:
            return ToolResult(error="timeout must be positive")
        limit = self.max_output_chars if max_output_chars is None else max(200, int(max_output_chars))
        style = self.truncation if truncation is None else truncation
        if style not in ("both", "head", "tail"):
            return ToolResult(error="truncation must be 'both', 'head' or 'tail'")

        if self.workspace is not None:
            # The workspace backend owns execution; cwd/env are not forwarded
            # there (callers run at the workspace root / preconfigured env).
            try:
                result = self.workspace.run_command(command, timeout=effective_timeout)
            except WorkspaceError as exc:
                return ToolResult(error=f"workspace command failed: {exc}")
            output, truncated_here = self._truncate(result.output, limit, style)
            workspace_metadata: dict[str, Any] = {
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "truncated": result.truncated or truncated_here,
                **result.metadata,
            }
            if result.error is not None:
                return ToolResult(output=output, error=result.error, metadata=workspace_metadata)
            if result.timed_out:
                return ToolResult(
                    output=output,
                    error=f"command timed out after {effective_timeout}s",
                    metadata=workspace_metadata,
                )
            if result.exit_code != 0:
                return ToolResult(
                    output=output,
                    error=f"command exited with code {result.exit_code}",
                    metadata=workspace_metadata,
                )
            return ToolResult(output=output, metadata=workspace_metadata)

        resolved_cwd: str | None = None
        if cwd is not None:
            resolved_cwd = os.path.abspath(cwd)
            if not os.path.isdir(resolved_cwd):
                return ToolResult(error=f"cwd is not a directory: {resolved_cwd!r}")
        call_env = self._merged_env(env)
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                [*self.command_wrapper, "bash", "-c", command],
                cwd=resolved_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,  # own process group -> kill the whole tree
                env=call_env,
            )
        except OSError as exc:
            return ToolResult(error=f"failed to start shell: {exc}")

        if self.progress_callback is not None:
            output, timed_out = self._run_with_progress(proc, effective_timeout)
        else:
            output, timed_out = self._communicate(proc, effective_timeout)

        duration = time.monotonic() - started
        output, truncated = self._truncate(output or "", limit, style)
        metadata: dict[str, Any] = {
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 3),
            "timed_out": timed_out,
            "truncated": truncated,
            "cwd": resolved_cwd,
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

    def _merged_env(self, call_env: dict[str, str] | None) -> dict[str, str] | None:
        """Combine tool-level and per-call env (`None` inherits the parent env)."""
        if self.env is None:
            return call_env  # None inherits os.environ; a dict replaces for this call
        if call_env is None:
            return self.env
        merged = dict(self.env)
        merged.update(call_env)
        return merged

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _communicate(self, proc: subprocess.Popen, timeout: float) -> tuple[str, bool]:
        """Collect all output via ``communicate`` (no streaming).

        After a timeout the process group is killed, but a child that called
        ``setsid`` may survive ``killpg`` and still hold the output pipe, so
        the follow-up ``communicate`` is itself bounded to avoid hanging past
        the promised wall-clock timeout.
        """
        try:
            output, _err = proc.communicate(timeout=timeout)
            return output or "", False
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            output, _err = self._collect_remaining(proc)
            return output or "", True

    @staticmethod
    def _collect_remaining(proc: subprocess.Popen) -> tuple[str, str]:
        """Drain the pipe with a bounded wait (never hang post-timeout)."""
        try:
            return proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
            try:
                return proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except OSError:
                        pass
                return "", ""

    def _run_with_progress(self, proc: subprocess.Popen, timeout: float) -> tuple[str, bool]:
        """Read output line-by-line, invoking ``progress_callback`` per line."""
        assert proc.stdout is not None  # we always spawn with stdout=PIPE
        import threading

        stream = proc.stdout
        buffer: list[str] = []

        def reader() -> None:
            # Text-mode iterator yields each line (with a trailing newline).
            for line in stream:
                buffer.append(line)
                stripped = line.rstrip("\n")
                if stripped and self.progress_callback is not None:
                    self.progress_callback(stripped)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_group(proc)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
        thread.join(timeout=2.0)
        if not timed_out:
            try:
                buffer.append(stream.read())
            except (OSError, ValueError):
                pass
        try:
            stream.close()
        except OSError:
            pass
        return "".join(buffer), timed_out

    def _truncate(self, text: str, limit: int, style: str) -> tuple[str, bool]:
        """Cut *text* to *limit* chars, keeping head/tail/both per *style*."""
        if len(text) <= limit:
            return text, False
        omitted = len(text) - limit
        marker = f"\n[... {omitted} ký tự ở giữa đã được cắt bớt ...]\n"
        if style == "head":
            return text[:limit] + marker, True
        if style == "tail":
            return marker + text[-limit:], True
        head = limit * 2 // 3
        tail = limit - head
        return text[:head] + marker + text[-tail:], True
