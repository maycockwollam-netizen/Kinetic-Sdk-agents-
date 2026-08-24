"""``FileTool``: view/create/edit files inside a :class:`Workspace`.

Every path goes through :meth:`Workspace.resolve`, so ``../``, absolute paths
outside the root and symlink escapes all fail before touching the disk — the
tool can never reach outside its workspace, whatever the model asks for.

Actions (one ``action`` parameter, mirroring the curated ``GitTool`` design):

* ``view`` — cat -n a file (optional ``view_range``), or list a directory one
  level deep.
* ``create`` — write a new file; refuses to overwrite an existing one.
* ``str_replace`` — replace a string that must match EXACTLY once.
* ``insert`` — insert text after a given 1-based line number.
* ``undo_edit`` — revert the last create/str_replace/insert on a path
  (single-level, in-memory backup; gone when the process exits).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, ClassVar

from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.manager import PathTraversalError, Workspace

logger = logging.getLogger(__name__)


class FileTool(Tool):
    """Workspace-scoped file editor with undo support.

    Args:
        workspace: The :class:`Workspace` all paths are resolved against.
            Required — a file tool without a workspace boundary is a
            whole-filesystem tool, which this class refuses to be.
        max_view_lines: Cap on lines returned by ``view`` (the rest is cut
            with a marker) so a huge file cannot flood the context.
    """

    name: str = "file_editor"
    description: str = (
        "View and edit text files inside the workspace. Actions: view (with "
        "optional view_range), create, str_replace (unique match), insert "
        "after a line, undo_edit. Paths are workspace-relative."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "description": "The editing operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the file (or directory for view).",
            },
            "file_text": {"type": "string", "description": "Content for create."},
            "old_str": {"type": "string", "description": "Text to replace (str_replace)."},
            "new_str": {"type": "string", "description": "Replacement / inserted text."},
            "insert_line": {
                "type": "integer",
                "description": "1-based line number to insert after (insert).",
            },
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional [start, end] 1-based line range for view; -1 = end of file.",
            },
        },
        "required": ["action", "path"],
    }

    DEFAULT_MAX_VIEW_LINES: ClassVar[int] = 2_000

    def __init__(
        self, workspace: Workspace, max_view_lines: int = DEFAULT_MAX_VIEW_LINES
    ) -> None:
        if max_view_lines < 10:
            raise ValueError("max_view_lines must be >= 10")
        self.workspace = workspace
        self.max_view_lines = max_view_lines
        # path -> previous content, for single-level undo_edit.
        self._undo: dict[Path, str | None] = {}

    # --- dispatch -----------------------------------------------------

    def execute(  # type: ignore[override]
        self, action: str, path: str, **params: Any
    ) -> ToolResult:
        # Named-parameter signature by design: the agent loop always invokes
        # tools via execute(**model_arguments) after schema validation, so
        # narrowing the base **params contract is safe here.
        try:
            target = Path(self.workspace.resolve(path))
        except PathTraversalError as exc:
            return ToolResult(error=f"path rejected: {exc}")

        actions: dict[str, Callable[..., ToolResult]] = {
            "view": self._view,
            "create": self._create,
            "str_replace": self._str_replace,
            "insert": self._insert,
            "undo_edit": self._undo_edit,
        }
        handler = actions.get(action)
        if handler is None:
            return ToolResult(
                error=f"unknown action {action!r} (expected view/create/str_replace/insert/undo_edit)"
            )
        try:
            return handler(target, **params)
        except OSError as exc:
            return ToolResult(error=f"filesystem error: {exc}")

    # --- actions --------------------------------------------------------

    def _view(self, target: Path, view_range: list[int] | None = None, **_: Any) -> ToolResult:
        if target.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult(output="\n".join(entries) or "(empty directory)")
        if not target.is_file():
            return ToolResult(error=f"no such file: {target.name}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = 1, len(lines)
        if view_range is not None:
            if len(view_range) != 2:
                return ToolResult(error="view_range must be [start, end]")
            start, end = view_range
            if end == -1:
                end = len(lines)
            if start < 1 or end < start:
                return ToolResult(error=f"invalid view_range {view_range}")
        sliced = lines[start - 1 : end]
        truncated = False
        if len(sliced) > self.max_view_lines:
            sliced = sliced[: self.max_view_lines]
            truncated = True
        numbered = "\n".join(
            f"{i}\t{line}" for i, line in enumerate(sliced, start=start)
        )
        if truncated:
            numbered += f"\n[... còn {end - start + 1 - self.max_view_lines} dòng nữa, bị cắt bớt ...]"
        return ToolResult(
            output=numbered,
            metadata={"total_lines": len(lines), "truncated": truncated},
        )

    def _create(self, target: Path, file_text: str | None = None, **_: Any) -> ToolResult:
        if file_text is None:
            return ToolResult(error="create requires file_text")
        if target.exists():
            return ToolResult(error=f"file already exists: {target.name} (use str_replace to edit)")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._undo[target] = None  # undo of a create = delete
        target.write_text(file_text, encoding="utf-8")
        return ToolResult(output=f"created {target.name}", metadata={"path": str(target)})

    def _str_replace(
        self,
        target: Path,
        old_str: str | None = None,
        new_str: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if old_str is None or new_str is None:
            return ToolResult(error="str_replace requires old_str and new_str")
        if not target.is_file():
            return ToolResult(error=f"no such file: {target.name}")
        content = target.read_text(encoding="utf-8", errors="replace")
        occurrences = content.count(old_str)
        if occurrences == 0:
            return ToolResult(error="old_str not found in the file")
        if occurrences > 1:
            return ToolResult(
                error=f"old_str matches {occurrences} times; it must match exactly once"
            )
        self._undo[target] = content
        target.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        return ToolResult(output=f"edited {target.name}", metadata={"path": str(target)})

    def _insert(
        self,
        target: Path,
        insert_line: int | None = None,
        new_str: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if insert_line is None or new_str is None:
            return ToolResult(error="insert requires insert_line and new_str")
        if not target.is_file():
            return ToolResult(error=f"no such file: {target.name}")
        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        if not 0 <= insert_line <= len(lines):
            return ToolResult(error=f"insert_line {insert_line} out of range 0..{len(lines)}")
        self._undo[target] = content
        new_lines = new_str.splitlines(keepends=True)
        if new_str and not new_str.endswith("\n"):
            new_lines[-1] += "\n"
        lines[insert_line:insert_line] = new_lines
        target.write_text("".join(lines), encoding="utf-8")
        return ToolResult(
            output=f"inserted into {target.name} after line {insert_line}",
            metadata={"path": str(target)},
        )

    def _undo_edit(self, target: Path, **_: Any) -> ToolResult:
        if target not in self._undo:
            return ToolResult(error=f"nothing to undo for {target.name}")
        previous = self._undo.pop(target)
        if previous is None:
            target.unlink(missing_ok=True)
            return ToolResult(output=f"undid create of {target.name}")
        current = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        target.write_text(previous, encoding="utf-8")
        self._undo[target] = current  # undo is itself undoable (redo)
        return ToolResult(output=f"reverted {target.name} to its previous content")
