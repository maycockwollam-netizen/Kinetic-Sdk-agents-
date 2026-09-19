"""Workspace-scoped content and filename search tools.

These tools intentionally require a local, resolving workspace.  They never
fall back to host-wide search: every input and every discovered path is passed
through ``workspace.resolve()`` before it is read or returned.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from kinetic_sdk.tool.base import (
    Tool,
    ToolCapability,
    ToolExecutionPolicy,
    ToolResult,
    ToolRiskLevel,
)
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError


class _WorkspaceSearchTool(Tool):
    """Shared local-workspace resolution guard for read-only search tools."""

    risk_level = ToolRiskLevel.READ_ONLY
    capabilities = frozenset({ToolCapability.FILESYSTEM})
    # The agent owns timeout and circuit-breaker handling.  In particular, do
    # not add a second subprocess timeout here: it would diverge from the
    # execution policy applied uniformly to every tool call.
    execution_policy = ToolExecutionPolicy(
        timeout_seconds=30.0,
        circuit_breaker_threshold=3,
    )

    def __init__(self, workspace: WorkspaceBase) -> None:
        self.workspace = workspace

    def _resolve_root(self, path: str) -> Path:
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
            # A symlink discovered beneath the root can still point outside.
            return None
        return Path(os.path.relpath(resolved, self.workspace.root_path)).as_posix()

    @staticmethod
    def _validate_max_results(value: int, default: int) -> int:
        max_results = default if value is None else value
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        return max_results


class GrepTool(_WorkspaceSearchTool):
    """Search file contents within a required workspace, never the whole system."""

    name = "grep"
    description = (
        "Search regex content in files inside the required workspace. Returns "
        "workspace-relative file:line:content matches."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string"},
            "case_sensitive": {"type": "boolean", "default": True},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["pattern"],
    }
    DEFAULT_MAX_RESULTS: ClassVar[int] = 100
    MAX_LINE_CHARS: ClassVar[int] = 500

    def execute(  # type: ignore[override]
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_sensitive: bool = True,
        max_results: int = DEFAULT_MAX_RESULTS,
        **_: Any,
    ) -> ToolResult:
        try:
            if not isinstance(pattern, str):
                return ToolResult(error="pattern must be a string")
            limit = self._validate_max_results(max_results, self.DEFAULT_MAX_RESULTS)
            root = self._resolve_root(path)  # mandatory workspace containment check
            if shutil.which("rg"):
                matches = self._grep_rg(pattern, root, glob, case_sensitive, limit)
            else:
                matches = self._grep_python(pattern, root, glob, case_sensitive, limit)
            return ToolResult(output="\n".join(matches) or "(no matches)", metadata={"count": len(matches)})
        except (OSError, WorkspaceError, ValueError, re.error) as exc:
            return ToolResult(error=f"search failed: {exc}")

    def _grep_rg(
        self, pattern: str, root: Path, glob: str | None, case_sensitive: bool, limit: int
    ) -> list[str]:
        command = ["rg", "--with-filename", "--line-number", "--no-heading", "--color", "never", "--max-count", str(limit)]
        if not case_sensitive:
            command.append("--ignore-case")
        if glob:
            command.extend(["--glob", glob])
        command.extend(["--", pattern, str(root)])
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        if completed.returncode not in (0, 1):
            raise WorkspaceError(completed.stderr.strip() or "ripgrep failed")
        matches: list[str] = []
        for raw in completed.stdout.splitlines():
            # Greedily capture the filename so normal absolute POSIX paths
            # and filenames containing colons remain valid.  Regardless of
            # formatting, _safe_relative performs the authoritative check.
            match = re.match(r"^(.*):(\d+):(.*)$", raw)
            if match is None:
                continue
            filename, line_number, content = match.groups()
            relative = self._safe_relative(Path(filename))
            if relative is not None:
                matches.append(f"{relative}:{line_number}:{self._truncate_line(content)}")
            if len(matches) >= limit:
                break
        return matches

    def _grep_python(
        self, pattern: str, root: Path, glob: str | None, case_sensitive: bool, limit: int
    ) -> list[str]:
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        files = [root] if root.is_file() else (Path(dirpath) / name for dirpath, _, names in os.walk(root, followlinks=False) for name in names)
        matches: list[str] = []
        for candidate in files:
            relative = self._safe_relative(candidate)
            if relative is None or (glob is not None and not fnmatch.fnmatch(relative, glob)):
                continue
            try:
                with candidate.open(errors="replace") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if regex.search(line):
                            matches.append(f"{relative}:{line_number}:{self._truncate_line(line.rstrip(chr(10)))}")
                            if len(matches) >= limit:
                                return matches
            except (IsADirectoryError, OSError):
                continue
        return matches

    @classmethod
    def _truncate_line(cls, line: str) -> str:
        return line if len(line) <= cls.MAX_LINE_CHARS else line[: cls.MAX_LINE_CHARS] + "...(cut)"


class GlobTool(_WorkspaceSearchTool):
    """Find filenames in a required workspace, never anywhere on the host system."""

    name = "glob"
    description = "Find files matching a glob pattern inside the required workspace."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "max_results": {"type": "integer", "default": 200},
        },
        "required": ["pattern"],
    }
    DEFAULT_MAX_RESULTS: ClassVar[int] = 200

    def execute(  # type: ignore[override]
        self, pattern: str, path: str = ".", max_results: int = DEFAULT_MAX_RESULTS, **_: Any
    ) -> ToolResult:
        try:
            if not isinstance(pattern, str):
                return ToolResult(error="pattern must be a string")
            limit = self._validate_max_results(max_results, self.DEFAULT_MAX_RESULTS)
            root = self._resolve_root(path)  # mandatory workspace containment check
            candidates = [root] if root.is_file() and root.match(pattern) else root.glob(pattern)
            matches: list[str] = []
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                relative = self._safe_relative(candidate)
                if relative is not None:
                    matches.append(relative)
                if len(matches) >= limit:
                    break
            matches.sort()
            return ToolResult(output="\n".join(matches) or "(no matches)", metadata={"count": len(matches)})
        except (OSError, WorkspaceError, ValueError) as exc:
            return ToolResult(error=f"search failed: {exc}")
