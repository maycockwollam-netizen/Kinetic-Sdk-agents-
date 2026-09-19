"""Atomic application of workspace-confined unified diffs.

The implementation intentionally uses only the standard library.  A patch is
parsed and applied to in-memory copies of every target before any workspace
write is attempted; this keeps the SDK core dependency-free and makes a
multi-file edit all-or-nothing in the normal case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from kinetic_sdk.subagent.filelock import FileLockRegistry
from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError


class PatchApplyError(ValueError):
    """Raised when a syntactically valid patch cannot be applied."""


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _FilePatch:
    path: str
    hunks: tuple[_Hunk, ...]


_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?:.*)$"
)


class ApplyPatchTool(Tool):
    """Apply a unified-diff patch to existing workspace files atomically.

    The ``patch`` argument must contain standard unified-diff file and hunk
    headers (``--- a/path``, ``+++ b/path``, and ``@@ ... @@``).  For example::

        --- a/src/example.py
        +++ b/src/example.py
        @@ -1 +1 @@
        -old_value = 1
        +new_value = 2

    Every target is resolved through the configured workspace.  All hunks are
    dry-run against in-memory content first; a context mismatch or unsafe path
    leaves every file untouched.  A rare write failure after that dry-run
    rolls back files already written.
    """

    name: str = "apply_patch"
    description: str = (
        "Atomically apply a multi-file unified diff inside the workspace. "
        "The patch must use --- a/path, +++ b/path, and @@ hunk headers."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": (
                    "Standard unified diff containing one or more files, with "
                    "--- a/path, +++ b/path, and @@ -old,+new @@ headers."
                ),
            }
        },
        "required": ["patch"],
    }

    def __init__(
        self,
        workspace: WorkspaceBase,
        *,
        lock_registry: FileLockRegistry | None = None,
        lock_timeout: float = 5.0,
    ) -> None:
        if lock_timeout < 0:
            raise ValueError("lock_timeout must be non-negative")
        self.workspace = workspace
        self._lock_registry = lock_registry
        self.lock_timeout = lock_timeout
        self._owner_id: str | None = None
        # Keep one previous value per path, matching FileTool's undo storage
        # shape.  This makes the pre-patch text available to callers that
        # coordinate editor undo state around tool instances.
        self._undo: dict[str, str] = {}

    def _write_locked(self, path: str, operation: Callable[[], None]) -> None:
        if self._lock_registry is None:
            operation()
            return
        if self._owner_id is None:
            raise ValueError(
                "ApplyPatchTool with lock_registry is not bound to an agent; "
                "set an owner before applying patches"
            )
        with self._lock_registry.acquire(path, self._owner_id, timeout=self.lock_timeout):
            operation()

    def execute(self, patch: str, **_: Any) -> ToolResult:  # type: ignore[override]
        """Dry-run *patch*, then apply it or leave the workspace unchanged."""
        try:
            file_patches = self._parse_patch(patch)
        except PatchApplyError as exc:
            return ToolResult(error=f"patch is empty or malformed: {exc}")
        if not file_patches:
            return ToolResult(error="patch is empty or malformed")

        originals: dict[str, str] = {}
        resolved: dict[str, str] = {}
        for file_patch in file_patches:
            path = file_patch.path
            try:
                # Always make the explicit boundary check before a backend
                # read, including backends whose read_text checks internally.
                self.workspace.resolve(path)
                original = self.workspace.read_text(path)
            except FileNotFoundError:
                return ToolResult(error=f"apply_patch aborted: {path} does not exist")
            except (ValueError, WorkspaceError) as exc:
                return ToolResult(error=f"apply_patch aborted: {path}: {exc}")
            except OSError as exc:
                return ToolResult(error=f"apply_patch aborted: {path}: {exc}")
            try:
                resolved[path] = self._apply_file_patch(original, file_patch)
            except PatchApplyError as exc:
                return ToolResult(error=f"apply_patch aborted: {path}: {exc}")
            originals[path] = original

        written: list[str] = []
        try:
            for path, new_content in resolved.items():
                def write(path: str = path, new_content: str = new_content) -> None:
                    self.workspace.write_text(path, new_content)

                self._write_locked(
                    path,
                    write,
                )
                written.append(path)
        except Exception as exc:
            rollback_errors: list[str] = []
            for path in reversed(written):
                try:
                    def rollback(path: str = path) -> None:
                        self.workspace.write_text(path, originals[path])

                    self._write_locked(
                        path,
                        rollback,
                    )
                except Exception as rollback_exc:  # pragma: no cover - catastrophic I/O
                    rollback_errors.append(f"{path}: {rollback_exc}")
            detail = f"apply_patch failed mid-write, rolled back: {exc}"
            if rollback_errors:
                detail += f" (rollback errors: {'; '.join(rollback_errors)})"
            return ToolResult(error=detail)

        self._undo.update(originals)
        return ToolResult(
            output=f"applied patch to {len(written)} file(s)", metadata={"files": written}
        )

    @staticmethod
    def _parse_path(header: str, prefix: str) -> str:
        value = header[len(prefix) :].split("\t", 1)[0].strip()
        if value.startswith(("a/", "b/")):
            value = value[2:]
        if not value or value == "/dev/null":
            raise PatchApplyError("patches must modify existing workspace files")
        return value

    @classmethod
    def _parse_patch(cls, patch: str) -> list[_FilePatch]:
        if not isinstance(patch, str) or not patch.strip():
            raise PatchApplyError("no patch content")
        lines = patch.splitlines(keepends=True)
        patches: list[_FilePatch] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ")):
                index += 1
                continue
            if not line.startswith("--- "):
                raise PatchApplyError(f"expected --- file header at line {index + 1}")
            old_path = cls._parse_path(line, "--- ")
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise PatchApplyError("missing +++ file header")
            new_path = cls._parse_path(lines[index], "+++ ")
            if old_path != new_path:
                raise PatchApplyError("renames are not supported")
            path = new_path
            index += 1
            hunks: list[_Hunk] = []
            while index < len(lines) and not lines[index].startswith("--- "):
                header = _HUNK_HEADER.match(lines[index].rstrip("\r\n"))
                if header is None:
                    if lines[index].startswith(("diff --git ", "index ")):
                        break
                    raise PatchApplyError(f"expected @@ hunk header at line {index + 1}")
                old_count = int(header.group("old_count") or "1")
                new_count = int(header.group("new_count") or "1")
                index += 1
                hunk_lines: list[str] = []
                old_seen = new_seen = 0
                while index < len(lines):
                    hunk_line = lines[index]
                    # A subsequent file header also begins with ``-`` / ``+``;
                    # recognise it before interpreting ordinary hunk lines.
                    if hunk_line.startswith("--- ") or hunk_line.startswith("diff --git "):
                        break
                    if hunk_line.startswith("\\ No newline at end of file"):
                        index += 1
                        continue
                    if hunk_line.startswith((" ", "+", "-")):
                        hunk_lines.append(hunk_line)
                        if hunk_line[0] in " -":
                            old_seen += 1
                        if hunk_line[0] in " +":
                            new_seen += 1
                        index += 1
                        continue
                    break
                if old_seen != old_count or new_seen != new_count:
                    raise PatchApplyError("hunk line counts do not match its header")
                hunks.append(
                    _Hunk(
                        int(header.group("old_start")), old_count,
                        int(header.group("new_start")), new_count, tuple(hunk_lines)
                    )
                )
            if not hunks:
                raise PatchApplyError(f"no hunks for {path}")
            patches.append(_FilePatch(path, tuple(hunks)))
        return patches

    @staticmethod
    def _apply_file_patch(original: str, file_patch: _FilePatch) -> str:
        lines = original.splitlines(keepends=True)
        offset = 0
        for hunk in file_patch.hunks:
            # Unified diffs use ``-0,0`` when inserting before the first
            # line, where subtracting one would point before the file.
            old_index = hunk.old_start if hunk.old_count == 0 else hunk.old_start - 1
            start = old_index + offset
            if start < 0 or start > len(lines):
                raise PatchApplyError("hunk position is outside the file")
            expected = [line[1:] for line in hunk.lines if line[0] in " -"]
            replacement = [line[1:] for line in hunk.lines if line[0] in " +"]
            end = start + len(expected)
            if lines[start:end] != expected:
                raise PatchApplyError("hunk context does not match the file")
            lines[start:end] = replacement
            offset += len(replacement) - len(expected)
        return "".join(lines)
