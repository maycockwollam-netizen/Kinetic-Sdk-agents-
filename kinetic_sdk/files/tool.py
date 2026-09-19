"""``FileTool``: view/create/edit files through a ``WorkspaceBase``.

Every operation is delegated to the workspace backend.  ``LocalWorkspace``
enforces realpath containment, while Docker and remote backends enforce their
own boundary at the execution site; the editor never reaches around a backend
with direct host filesystem access.

Actions (one ``action`` parameter, mirroring the curated ``GitTool`` design):

* ``view`` — cat -n a file (optional ``view_range``), or list a directory one
  level deep.
* ``create`` — write a new file; refuses to overwrite an existing one.
* ``str_replace`` — replace a uniquely matched string, with a conservative
  whitespace/anchored-block fallback; exact matches may be replaced all at once.
* ``insert`` — insert text after a given 1-based line number.
* ``undo_edit`` — revert the last create/str_replace/insert on a path
  (single-level, in-memory backup; gone when the process exits).
"""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from kinetic_sdk.files.snapshots import (
    GitSnapshotStore,
    SnapshotStoreError,
    is_internal_snapshot_path,
)
from kinetic_sdk.subagent.exceptions import FileLockTimeoutError
from kinetic_sdk.subagent.filelock import FileLockRegistry
from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.agent import Agent

logger = logging.getLogger(__name__)


def _line_contents_and_spans(text: str) -> list[tuple[str, int, int, int]]:
    """Return physical-line text plus its start, body-end, and full-end offsets."""
    result: list[tuple[str, int, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        body_end = offset + len(body)
        result.append((body, offset, body_end, offset + len(line)))
        offset += len(line)
    # ``splitlines`` intentionally has no item for an empty string.  This is
    # useful here: fuzzy matching an empty replacement target is unsafe.
    return result


def _normalise_indentation(line: str) -> str:
    """Represent every non-empty leading space/tab run by one indentation unit."""
    index = 0
    while index < len(line) and line[index] in " \t":
        index += 1
    return (" " if index else "") + line[index:]


def _preserve_newline_style(value: str, reference: str) -> str:
    """Use CRLF for inserted text when editing a CRLF file."""
    if "\r\n" not in reference:
        return value
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")


def _fuzzy_find_details(content: str, old_str: str) -> tuple[str | None, list[tuple[int, int]]]:
    """Find one conservative non-exact match and report its strategy.

    Strategies become progressively looser.  A strategy only wins when it
    produces exactly one candidate; otherwise the next strategy is tried.
    """
    content_lines = _line_contents_and_spans(content)
    old_lines = _line_contents_and_spans(old_str)
    if not content_lines or not old_lines:
        return None, []

    old_bodies = [line[0] for line in old_lines]
    include_final_newline = old_str.endswith(("\n", "\r"))

    def contiguous_matches(transform: Callable[[str], str]) -> list[tuple[int, int]]:
        if len(old_bodies) > len(content_lines):
            return []
        wanted = [transform(line) for line in old_bodies]
        matches: list[tuple[int, int]] = []
        for start_index in range(len(content_lines) - len(wanted) + 1):
            candidate = [
                transform(line[0])
                for line in content_lines[start_index : start_index + len(wanted)]
            ]
            if candidate == wanted:
                last = content_lines[start_index + len(wanted) - 1]
                matches.append((content_lines[start_index][1], last[3] if include_final_newline else last[2]))
        return matches

    strategies: tuple[tuple[str, Callable[[str], str]], ...] = (
        # Keep tabs intact at this stage: tab-vs-space indentation belongs to
        # the next, explicitly reported indentation-normalisation strategy.
        ("line_whitespace_trimmed", lambda line: line.strip(" ")),
        ("indentation_normalized", _normalise_indentation),
    )
    for name, transform in strategies:
        matches = contiguous_matches(transform)
        if len(matches) == 1:
            return name, matches

    # Anchors are only a recovery mechanism, not permission to replace an
    # arbitrarily large interval.  A candidate must remain close in length and
    # retain substantial middle content.
    if len(old_bodies) >= 3:
        matches = []
        rejected = False
        allowed_delta = max(2, int(len(old_bodies) * 0.2))
        for first_index, first in enumerate(content_lines):
            if first[0] != old_bodies[0]:
                continue
            for last_index in range(first_index + 1, len(content_lines)):
                last = content_lines[last_index]
                if last[0] == old_bodies[-1]:
                    candidate_bodies = [line[0] for line in content_lines[first_index : last_index + 1]]
                    if abs(len(candidate_bodies) - len(old_bodies)) > allowed_delta:
                        rejected = True
                        continue
                    ratio = difflib.SequenceMatcher(
                        None, "\n".join(old_bodies), "\n".join(candidate_bodies)
                    ).ratio()
                    if ratio < 0.6:
                        rejected = True
                        continue
                    matches.append((first[1], last[3] if include_final_newline else last[2]))
        if len(matches) == 1:
            return "anchored_block", matches
        if rejected and not matches:
            return "anchored_block_rejected", []
        return None, matches
    return None, []


def _reindent_fuzzy_replacement(old_str: str, new_str: str, matched: str) -> str | None:
    """Map replacement indentation onto the indentation used by *matched*.

    Fuzzy indentation matching proves that the model used a different
    whitespace convention.  Writing its text verbatim would create mixed
    indentation, so use corresponding matched lines when possible and reject
    an ambiguous style rather than guessing.
    """
    old_lines = old_str.splitlines(keepends=True)
    new_lines = new_str.splitlines(keepends=True)
    matched_lines = matched.splitlines(keepends=True)
    if not new_lines:
        return new_str

    def indent(line: str) -> str:
        return line[: len(line) - len(line.lstrip(" \t"))]

    target_indents = [indent(line) for line in matched_lines]
    if not any(target_indents):
        return new_str if not any(indent(line) for line in new_lines) else None
    uses_tabs = any("\t" in value for value in target_indents)
    if uses_tabs and any(" " in value for value in target_indents if value):
        return None
    if len(new_lines) != len(old_lines) or len(matched_lines) != len(old_lines):
        return None

    result: list[str] = []
    for old_line, new_line, target_indent in zip(old_lines, new_lines, target_indents):
        old_indent = indent(old_line)
        new_indent = indent(new_line)
        if not new_indent:
            result.append(new_line)
            continue
        if new_indent == old_indent:
            result.append(target_indent + new_line[len(new_indent) :])
            continue
        # A changed relative depth is only safe when the target convention is
        # unambiguous.  Four spaces is the conventional source indentation.
        if uses_tabs:
            depth = len(new_indent.expandtabs(4)) // 4
            result.append("\t" * depth + new_line[len(new_indent) :])
        else:
            widths = [len(value) for value in target_indents if value]
            width = min(widths) if widths else 0
            if width <= 0:
                return None
            depth = len(new_indent) // 4
            result.append(" " * (width * depth) + new_line[len(new_indent) :])
    return "".join(result)


def _fuzzy_find(content: str, old_str: str) -> list[tuple[int, int]]:
    """Return conservative fuzzy-match spans, or no/ambiguous spans.

    Exact matching remains the responsibility of :meth:`FileTool._str_replace`
    and is always attempted first.
    """
    _, matches = _fuzzy_find_details(content, old_str)
    return matches


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
        snapshot_store: Optional private ``GitSnapshotStore`` that records the
            content before each ``str_replace`` and ``insert``. ``None``
            (the default) makes no snapshot files or repositories.
    """

    name: str = "file_editor"
    description: str = (
        "View and edit text files inside the workspace. Actions: view (with "
        "optional view_range), create, str_replace (exact match preferred; "
        "conservative fuzzy fallback; optional replace_all for exact matches), "
        "insert after a line, undo_edit. Paths are workspace-relative."
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
            "old_str": {
                "type": "string",
                "description": (
                    "Text to replace (str_replace). Exact matching is preferred; "
                    "a unique whitespace/indentation/anchored-block fallback may apply."
                ),
            },
            "new_str": {"type": "string", "description": "Replacement / inserted text."},
            "replace_all": {
                "type": "boolean",
                "description": (
                    "Replace every exact old_str match (str_replace only). Defaults "
                    "to false and cannot be combined with fuzzy matching."
                ),
            },
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
    MAX_VIEW_LINE_CHARS: ClassVar[int] = 500

    def __init__(
        self,
        workspace: WorkspaceBase,
        max_view_lines: int = DEFAULT_MAX_VIEW_LINES,
        *,
        lock_registry: FileLockRegistry | None = None,
        lock_timeout: float = 5.0,
        snapshot_store: GitSnapshotStore | None = None,
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
        self._snapshot_store = snapshot_store
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
            snapshot_store=self._snapshot_store,
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
            if is_internal_snapshot_path(path):
                return ToolResult(error="path is reserved for internal snapshot storage")
            return handler(path, **params)
        except FileLockTimeoutError as exc:
            logger.warning("Write lock unavailable for %r: %s", path, exc)
            return ToolResult(error=f"file is locked for writing: {exc}")
        except ValueError as exc:
            return ToolResult(error=f"path rejected: {exc}")
        except (OSError, WorkspaceError, SnapshotStoreError) as exc:
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
            f"{i}\t{self._truncate_view_line(line)}"
            for i, line in enumerate(sliced, start=start)
        )
        if truncated:
            remaining = len(lines[start - 1 : end]) - len(sliced)
            next_start = start + len(sliced)
            numbered += (
                f"\n[... còn {remaining} dòng nữa; dùng "
                f"view_range=[{next_start}, {end}] để xem tiếp ...]"
            )
        return ToolResult(
            output=numbered,
            metadata={"total_lines": len(lines), "truncated": truncated},
        )

    @classmethod
    def _truncate_view_line(cls, line: str) -> str:
        """Cap one physical line so a minified/log line cannot flood context."""
        if len(line) <= cls.MAX_VIEW_LINE_CHARS:
            return line
        return line[: cls.MAX_VIEW_LINE_CHARS] + "...(cut)"

    def _view_directory(self, path: str) -> ToolResult:
        """View one portable workspace directory level."""
        try:
            entries = self.workspace.list_directory(path)
        except (FileNotFoundError, NotADirectoryError):
            return ToolResult(error=f"no such file or directory: {path}")
        entries = [entry for entry in entries if entry.rstrip("/") != ".kinetic"]
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
        *,
        replace_all: bool = False,
        **_: Any,
    ) -> ToolResult:
        if old_str is None or new_str is None:
            return ToolResult(error="str_replace requires old_str and new_str")
        try:
            content = self.workspace.read_text(path)
        except FileNotFoundError:
            return ToolResult(error=f"no such file: {path}")
        occurrences = content.count(old_str)
        if occurrences > 1 and not replace_all:
            return ToolResult(
                error=(
                    f"old_str matches {occurrences} times; pass replace_all=True to "
                    "replace all of them, or narrow old_str to match exactly once"
                )
            )
        if occurrences:
            count = occurrences if replace_all else 1
            new_content = content.replace(old_str, _preserve_newline_style(new_str, content), count)
            metadata: dict[str, Any] = {"path": path, "count": count}
        else:
            strategy, matches = _fuzzy_find_details(content, old_str)
            if strategy == "anchored_block_rejected":
                return ToolResult(
                    error=(
                        "anchored_block fuzzy match rejected because the matched block "
                        "is too different in size or middle content; use a more exact old_str"
                    )
                )
            if not matches:
                return ToolResult(error="old_str not found in the file")
            if len(matches) > 1:
                return ToolResult(
                    error=(
                        f"old_str fuzzy-matches {len(matches)} times; it must match "
                        "exactly once"
                    )
                )
            if replace_all:
                return ToolResult(
                    error=(
                        "replace_all=True only supports exact matches; narrow old_str "
                        "or disable replace_all to use fuzzy matching"
                    )
                )
            start, end = matches[0]
            replacement = _preserve_newline_style(new_str, content)
            if strategy == "indentation_normalized":
                reindented = _reindent_fuzzy_replacement(
                    old_str, replacement, content[start:end]
                )
                if reindented is None:
                    return ToolResult(
                        error=(
                            "indentation_normalized fuzzy match rejected because replacement "
                            "indentation could not be inferred safely"
                        )
                    )
                replacement = reindented
            new_content = content[:start] + replacement + content[end:]
            count = 1
            start_line = content.count("\n", 0, start) + 1
            end_line = content.count("\n", 0, end) + (0 if end > start and content[end - 1 : end] == "\n" else 1)
            metadata = {
                "path": path,
                "count": count,
                "match_strategy": strategy,
                "lines": f"{start_line}-{end_line}",
            }
        self._write_locked(
            path,
            lambda: self._snapshot_and_write(path, content, new_content),
        )
        self._undo[path] = content
        if "match_strategy" in metadata:
            return ToolResult(
                output=(f"edited {path} ({count} replacement, "
                        f"fuzzy={metadata['match_strategy']}, lines {metadata['lines']})"),
                metadata=metadata,
            )
        return ToolResult(output=f"edited {path} ({count} replacement(s))", metadata=metadata)

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
        new_str = _preserve_newline_style(new_str, content)
        new_lines = new_str.splitlines(keepends=True)
        if new_str and not new_str.endswith(("\n", "\r")):
            new_lines[-1] += "\n"
        lines[insert_line:insert_line] = new_lines
        self._write_locked(
            path, lambda: self._snapshot_and_write(path, content, "".join(lines))
        )
        self._undo[path] = content
        return ToolResult(
            output=f"inserted into {path} after line {insert_line}",
            metadata={"path": path},
        )

    def _snapshot_and_write(self, path: str, previous: str, updated: str) -> None:
        """Persist the pre-edit content immediately before the coordinated write."""
        if self._snapshot_store is not None:
            self._snapshot_store.snapshot(path, previous)
        self.workspace.write_text(path, updated)

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
