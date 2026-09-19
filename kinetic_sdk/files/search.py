"""Workspace-scoped content and filename search tools.

These tools intentionally require a local, resolving workspace.  They never
fall back to host-wide search: every input and every discovered path is passed
through ``workspace.resolve()`` before it is read or returned.
"""

from __future__ import annotations

import fnmatch
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

from kinetic_sdk.files.snapshots import is_internal_snapshot_path
from kinetic_sdk.tool.base import (
    Tool,
    ToolCapability,
    ToolExecutionPolicy,
    ToolResult,
    ToolRiskLevel,
)
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError


class _SearchTimeoutError(WorkspaceError):
    """Raised when a search engine does not finish within its hard deadline."""


def _python_grep_worker(
    result_queue: multiprocessing.queues.Queue[Any],
    root: str,
    pattern: str,
    glob: str | None,
    case_sensitive: bool,
    limit: int,
) -> None:
    """Run regex search in an isolated process so pathological regexes are killable."""
    try:
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        root_path = Path(root)
        candidates: list[Path] = []
        if root_path.is_file():
            candidates.append(root_path)
        else:
            for dirpath, directories, names in os.walk(root_path, followlinks=False):
                directories[:] = sorted(name for name in directories if not name.startswith("."))
                candidates.extend(Path(dirpath) / name for name in sorted(names) if not name.startswith("."))

        matches: list[tuple[str, int, str]] = []
        for candidate in candidates:
            try:
                if candidate.stat().st_size > GrepTool.MAX_FILE_BYTES:
                    continue
                with candidate.open("rb") as handle:
                    data = handle.read()
            except (IsADirectoryError, OSError):
                continue
            if b"\0" in data[: GrepTool.BINARY_CHECK_BYTES]:
                continue
            relative = candidate.relative_to(root_path).as_posix() if root_path.is_dir() else candidate.name
            if glob is not None and not fnmatch.fnmatch(relative, glob):
                continue
            for line_number, line in enumerate(data.decode(errors="replace").splitlines(), start=1):
                if regex.search(line):
                    matches.append((str(candidate), line_number, line))
                    if len(matches) >= limit:
                        result_queue.put(("ok", matches))
                        return
        result_queue.put(("ok", matches))
    except (OSError, re.error, ValueError) as exc:
        result_queue.put(("error", str(exc)))


class _WorkspaceSearchTool(Tool):
    """Shared local-workspace resolution guard for read-only search tools."""

    risk_level = ToolRiskLevel.READ_ONLY
    capabilities = frozenset({ToolCapability.FILESYSTEM})
    execution_policy = ToolExecutionPolicy(timeout_seconds=30.0, circuit_breaker_threshold=3)

    def __init__(self, workspace: WorkspaceBase) -> None:
        self.workspace = workspace

    def _resolve_root(self, path: str) -> Path:
        if is_internal_snapshot_path(path):
            raise WorkspaceError("path is reserved for internal snapshot storage")
        resolved = self.workspace.resolve(path)
        root = Path(resolved)
        if not root.exists():
            raise FileNotFoundError(path)
        return root

    def _safe_relative(self, candidate: Path) -> str | None:
        """Return a workspace-relative path only after a containment check."""
        try:
            resolved = Path(self.workspace.resolve(str(candidate)))
        except (ValueError, WorkspaceError):
            return None
        return Path(os.path.relpath(resolved, self.workspace.root_path)).as_posix()

    @staticmethod
    def _validate_max_results(value: int, default: int) -> int:
        max_results = default if value is None else value
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        return max_results


class GrepTool(_WorkspaceSearchTool):
    """Search contents in a workspace using rg or a bounded Python fallback.

    The Python fallback skips hidden files/directories, binary files, and files
    larger than 2 MB to approximate ripgrep's default safety behaviour.  It
    intentionally does *not* apply ``.gitignore`` rules because the fallback
    has no third-party ignore-file parser.
    """

    name = "grep"
    description = "Search regex content in files inside the required workspace."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string", "default": "."},
            "glob": {"type": "string"}, "case_sensitive": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["pattern"],
    }
    DEFAULT_MAX_RESULTS: ClassVar[int] = 100
    MAX_LINE_CHARS: ClassVar[int] = 500
    SEARCH_TIMEOUT_SECONDS: ClassVar[float] = 20.0
    RG_TIMEOUT_MARGIN_SECONDS: ClassVar[float] = 1.0
    MAX_RG_OUTPUT_BYTES: ClassVar[int] = 1_000_000
    MAX_FILE_BYTES: ClassVar[int] = 2 * 1024 * 1024
    BINARY_CHECK_BYTES: ClassVar[int] = 8 * 1024

    def execute(  # type: ignore[override]
        self, pattern: str, path: str = ".", glob: str | None = None,
        case_sensitive: bool = True, max_results: int = DEFAULT_MAX_RESULTS, **_: Any,
    ) -> ToolResult:
        try:
            if not isinstance(pattern, str):
                return ToolResult(error="pattern must be a string")
            limit = self._validate_max_results(max_results, self.DEFAULT_MAX_RESULTS)
            root = self._resolve_root(path)
            if shutil.which("rg"):
                matches = self._grep_rg(pattern, root, glob, case_sensitive, limit)
                engine = "rg"
            else:
                matches = self._grep_python(pattern, root, glob, case_sensitive, limit)
                engine = "python"
            return ToolResult(output="\n".join(matches) or "(no matches)", metadata={"count": len(matches), "engine": engine})
        except _SearchTimeoutError as exc:
            return ToolResult(error=f"regex timed out: {exc}")
        except (OSError, WorkspaceError, ValueError, re.error) as exc:
            return ToolResult(error=f"search failed: {exc}")

    def _grep_rg(self, pattern: str, root: Path, glob: str | None, case_sensitive: bool, limit: int) -> list[str]:
        command = ["rg", "--with-filename", "--line-number", "--no-heading", "--color", "never"]
        if not case_sensitive:
            command.append("--ignore-case")
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--", pattern, str(root)])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, errors="replace", start_new_session=True)
        lines: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            for raw in process.stdout:
                lines.put(raw)
            lines.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        policy_timeout = self.execution_policy.timeout_seconds or self.SEARCH_TIMEOUT_SECONDS
        deadline = time.monotonic() + min(
            self.SEARCH_TIMEOUT_SECONDS,
            max(0.1, policy_timeout - self.RG_TIMEOUT_MARGIN_SECONDS),
        )
        bytes_read = 0
        matches: list[str] = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _SearchTimeoutError("ripgrep exceeded its deadline")
                try:
                    raw = lines.get(timeout=remaining)
                except queue.Empty as exc:
                    raise _SearchTimeoutError("ripgrep exceeded its deadline") from exc
                if raw is None:
                    break
                bytes_read += len(raw.encode("utf-8", errors="replace"))
                if bytes_read > self.MAX_RG_OUTPUT_BYTES:
                    raise WorkspaceError("ripgrep output exceeded byte limit")
                match = re.match(r"^(.*):(\d+):(.*)$", raw.rstrip("\n"))
                if match is None:
                    continue
                filename, line_number, content = match.groups()
                relative = self._safe_relative(Path(filename))
                if relative is not None:
                    matches.append(f"{relative}:{line_number}:{self._truncate_line(content)}")
                if len(matches) >= limit:
                    break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            returncode = process.wait()
        if returncode not in (0, 1, -signal.SIGTERM):
            raise WorkspaceError("ripgrep failed")
        return matches

    def _grep_python(self, pattern: str, root: Path, glob: str | None, case_sensitive: bool, limit: int) -> list[str]:
        result_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue(maxsize=1)
        worker = multiprocessing.Process(target=_python_grep_worker, args=(result_queue, str(root), pattern, glob, case_sensitive, limit))
        worker.start()
        worker.join(self.SEARCH_TIMEOUT_SECONDS)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            raise _SearchTimeoutError("Python regex exceeded its deadline")
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            raise WorkspaceError("Python search worker exited without a result") from None
        if status == "error":
            raise WorkspaceError(str(payload))
        matches: list[str] = []
        for filename, line_number, content in payload:
            relative = self._safe_relative(Path(filename))
            if relative is not None:
                matches.append(f"{relative}:{line_number}:{self._truncate_line(content)}")
        return matches

    @classmethod
    def _truncate_line(cls, line: str) -> str:
        return line if len(line) <= cls.MAX_LINE_CHARS else line[: cls.MAX_LINE_CHARS] + "...(cut)"


class GlobTool(_WorkspaceSearchTool):
    """Find filenames in a required workspace, never anywhere on the host system."""

    name = "glob"
    description = "Find files matching a glob pattern inside the required workspace."
    parameters: dict[str, Any] = {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}, "max_results": {"type": "integer", "default": 200}}, "required": ["pattern"]}
    DEFAULT_MAX_RESULTS: ClassVar[int] = 200

    def execute(  # type: ignore[override]
        self, pattern: str, path: str = ".", max_results: int = DEFAULT_MAX_RESULTS, **_: Any,
    ) -> ToolResult:
        try:
            if not isinstance(pattern, str):
                return ToolResult(error="pattern must be a string")
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                return ToolResult(error="pattern must be relative to the workspace")
            if ".." in pattern_path.parts:
                return ToolResult(error="pattern must not contain '..'")
            limit = self._validate_max_results(max_results, self.DEFAULT_MAX_RESULTS)
            root = self._resolve_root(path)
            candidates = [root] if root.is_file() and root.match(pattern) else list(root.glob(pattern))
            candidates = [candidate for candidate in candidates if not is_internal_snapshot_path(str(candidate.relative_to(Path(self.workspace.root_path))))]
            matches = sorted(
                relative for candidate in candidates if candidate.is_file()
                for relative in [self._safe_relative(candidate)] if relative is not None
            )[:limit]
            return ToolResult(output="\n".join(matches) or "(no matches)", metadata={"count": len(matches)})
        except (OSError, WorkspaceError, ValueError) as exc:
            return ToolResult(error=f"search failed: {exc}")
