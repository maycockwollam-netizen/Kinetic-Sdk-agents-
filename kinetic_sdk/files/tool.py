"""``FileTool``: view/create/edit files through a ``WorkspaceBase``.

Every operation is delegated to the workspace backend.  ``LocalWorkspace``
enforces realpath containment, while Docker and remote backends enforce their
own boundary at the execution site; the editor never reaches around a backend
with direct host filesystem access.

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
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from kinetic_sdk.subagent.exceptions import FileLockTimeoutError
from kinetic_sdk.subagent.filelock import FileLockRegistry
from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.agent import Agent

logger = logging.getLogger(__name__)


class FileTool(Tool):
    """Workspace-scoped file editor with undo support.

    Args:
        workspace: The :class:`WorkspaceBase` all paths are resolved against.
            Required — a file tool without a workspace boundary is a
            whole-filesystem tool, which this class refuses to be.
        max_view_lines: Cap on lines returned by ``view`` (the rest is cut
            with a marker) so a huge file cannot flood the context.
        lock_registry: Optional shared sub-agent file-lock registry. When
            supplied, write actions acquire a short-lived per-file lease.
            Call :meth:`bind` after constructing the owning agent so the
            lease owner uses its stable delegation audit id.
        lock_timeout: Seconds to wait for another agent's write lease.
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
        self,
        workspace: WorkspaceBase,
        max_view_lines: int = DEFAULT_MAX_VIEW_LINES,
        *,
        lock_registry: FileLockRegistry | None = None,
        lock_timeout: float = 5.0,
    ) -> None:
        if max_view_lines < 10:
            raise ValueError("max_view_lines must be >= 10")
        if lock_timeout < 0:
            raise ValueError("lock_timeout must be non-negative")
        self.workspace = workspace
        self.max_view_lines = max_view_lines
        self._lock_registry = lock_registry
        self.lock_timeout = lock_timeout
        self._owner_id: str | None = None
        # path -> previous content, for single-level undo_edit.
        self._undo: dict[str, str | None] = {}

    def bind(self, agent: "Agent", *, owner_id: str | None = None) -> None:
        """Bind lock ownership to *agent*'s stable delegation audit id.

        Binding is only needed when ``lock_registry`` is configured. It is
        intentionally separate from construction, matching ``DelegateTool``
        and allowing the agent to be created with this tool in its set.
        """
        if owner_id is None:
            from kinetic_sdk.subagent.delegation import agent_id_for

            owner_id = agent_id_for(agent)
        self._owner_id = owner_id

    def _clone_for(self) -> "FileTool":
        """Return an unbound child tool sharing this registry and workspace."""
        return FileTool(
            self.workspace,
            self.max_view_lines,
            lock_registry=self._lock_registry,
            lock_timeout=self.lock_timeout,
        )

    # --- dispatch -----------------------------------------------------

    def execute(  # type: ignore[override]
        self, action: str, path: str, **params: Any
    ) -> ToolResult:
        # Named-parameter signature by design: the agent loop always invokes
        # tools via execute(**model_arguments) after schema validation, so
        # narrowing the base **params contract is safe here.
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
            return handler(path, **params)
        except FileLockTimeoutError as exc:
            logger.warning("Write lock unavailable for %r: %s", path, exc)
            return ToolResult(error=f"file is locked for writing: {exc}")
        except ValueError as exc:
            return ToolResult(error=f"path rejected: {exc}")
        except (OSError, WorkspaceError) as exc:
            return ToolResult(error=f"filesystem error: {exc}")

    # --- actions --------------------------------------------------------

    def _write_locked(self, path: str, operation: Callable[[], None]) -> None:
        """Run one backend write under this call's short-lived file lease."""
        if self._lock_registry is None:
            operation()
            return
        if self._owner_id is None:
            raise ValueError(
                "FileTool with lock_registry is not bound to an agent; "
                "call file_tool.bind(agent) after constructing the agent"
            )
        with self._lock_registry.acquire(
            path, self._owner_id, timeout=self.lock_timeout
        ):
            operation()

    def _view(self, path: str, view_range: list[int] | None = None, **_: Any) -> ToolResult:
        try:
            content = self.workspace.read_text(path)
        except (FileNotFoundError, IsADirectoryError):
            return self._view_directory(path)
        lines = content.splitlines()
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

    def _view_directory(self, path: str) -> ToolResult:
        """View one portable workspace directory level."""
        try:
            entries = self.workspace.list_directory(path)
        except (FileNotFoundError, NotADirectoryError):
            return ToolResult(error=f"no such file or directory: {path}")
        return ToolResult(output="\n".join(entries) or "(empty directory)")

    def _create(self, path: str, file_text: str | None = None, **_: Any) -> ToolResult:
        if file_text is None:
            return ToolResult(error="create requires file_text")
        try:
            self.workspace.read_text(path)
        except FileNotFoundError:
            pass
        else:
            return ToolResult(error=f"file already exists: {path} (use str_replace to edit)")
        self._write_locked(path, lambda: self.workspace.write_text(path, file_text))
        self._undo[path] = None  # undo of a create = delete
        return ToolResult(output=f"created {path}", metadata={"path": path})

    def _str_replace(
        self,
        path: str,
        old_str: str | None = None,
        new_str: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if old_str is None or new_str is None:
            return ToolResult(error="str_replace requires old_str and new_str")
        try:
            content = self.workspace.read_text(path)
        except FileNotFoundError:
            return ToolResult(error=f"no such file: {path}")
        occurrences = content.count(old_str)
        if occurrences == 0:
            return ToolResult(error="old_str not found in the file")
        if occurrences > 1:
            return ToolResult(
                error=f"old_str matches {occurrences} times; it must match exactly once"
            )
        self._write_locked(
            path,
            lambda: self.workspace.write_text(path, content.replace(old_str, new_str, 1)),
        )
        self._undo[path] = content
        return ToolResult(output=f"edited {path}", metadata={"path": path})

    def _insert(
        self,
        path: str,
        insert_line: int | None = None,
        new_str: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if insert_line is None or new_str is None:
            return ToolResult(error="insert requires insert_line and new_str")
        try:
            content = self.workspace.read_text(path)
        except FileNotFoundError:
            return ToolResult(error=f"no such file: {path}")
        lines = content.splitlines(keepends=True)
        if not 0 <= insert_line <= len(lines):
            return ToolResult(error=f"insert_line {insert_line} out of range 0..{len(lines)}")
        new_lines = new_str.splitlines(keepends=True)
        if new_str and not new_str.endswith("\n"):
            new_lines[-1] += "\n"
        lines[insert_line:insert_line] = new_lines
        self._write_locked(path, lambda: self.workspace.write_text(path, "".join(lines)))
        self._undo[path] = content
        return ToolResult(
            output=f"inserted into {path} after line {insert_line}",
            metadata={"path": path},
        )

    def _undo_edit(self, path: str, **_: Any) -> ToolResult:
        if path not in self._undo:
            return ToolResult(error=f"nothing to undo for {path}")
        previous = self._undo[path]
        if previous is None:
            self._write_locked(path, lambda: self.workspace.delete_file(path))
            self._undo.pop(path)
            return ToolResult(output=f"undid create of {path}")
        try:
            current = self.workspace.read_text(path)
        except FileNotFoundError:
            current = ""
        self._write_locked(path, lambda: self.workspace.write_text(path, previous))
        self._undo[path] = current  # undo is itself undoable (redo)
        return ToolResult(output=f"reverted {path} to its previous content")
