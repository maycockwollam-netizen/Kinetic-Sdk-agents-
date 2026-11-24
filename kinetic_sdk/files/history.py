"""Agent-facing access to opt-in durable file snapshot history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kinetic_sdk.files.snapshots import (
    GitSnapshotStore,
    SnapshotStoreError,
    is_internal_snapshot_path,
)
from kinetic_sdk.subagent.exceptions import FileLockTimeoutError
from kinetic_sdk.subagent.filelock import FileLockRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.tool.base import (
    Tool,
    ToolCapability,
    ToolExecutionPolicy,
    ToolResult,
    ToolRiskLevel,
)
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError


class FileHistoryTool(Tool):
    """List or restore private snapshots; restore overwrites a file and is WRITE-risk.

    Construct this only with the same :class:`GitSnapshotStore` supplied to a
    ``FileTool``. Restoring a snapshot replaces current file content, so
    production permission policies should require human confirmation for its
    ``restore`` action.
    """

    name = "file_history"
    description = "List durable file snapshots or restore a selected snapshot."
    risk_level = ToolRiskLevel.WRITE
    capabilities = frozenset({ToolCapability.FILESYSTEM})
    execution_policy = ToolExecutionPolicy()
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "restore"]},
            "path": {"type": "string"},
            "snapshot_id": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["action", "path"],
    }

    def __init__(
        self,
        workspace: WorkspaceBase,
        snapshot_store: GitSnapshotStore | None = None,
        lock_registry: FileLockRegistry | None = None,
        lock_timeout: float = 5.0,
    ) -> None:
        self._workspace = workspace
        self._snapshot_store = snapshot_store
        #: Shared per-file lock registry (same one FileTool/ApplyPatchTool use)
        #: so a concurrent restore cannot race another agent's write to the
        #: same file. ``None`` disables locking (historical behaviour).
        self._lock_registry = lock_registry
        self._lock_timeout = lock_timeout
        self._owner_id: str | None = None

    def bind(self, agent: "Agent", *, owner_id: str | None = None) -> None:
        """Bind lock ownership to *agent*'s stable delegation audit id.

        Required only when ``lock_registry`` is configured, mirroring
        :meth:`FileTool.bind` so sub-agents share the cross-tree lock registry.
        """
        if owner_id is None:
            from kinetic_sdk.subagent.delegation import agent_id_for

            owner_id = agent_id_for(agent)
        self._owner_id = owner_id

    def _write_locked(self, path: str, operation: Any) -> Any:
        """Run one backend write under the shared per-file lease."""
        if self._lock_registry is None:
            return operation()
        if self._owner_id is None:
            raise ValueError(
                "FileHistoryTool with lock_registry is not bound to an agent; "
                "call file_history_tool.bind(agent) after constructing the agent"
            )
        with self._lock_registry.acquire(path, self._owner_id, timeout=self._lock_timeout):
            return operation()

    def execute(self, action: str, path: str, **params: Any) -> ToolResult:  # type: ignore[override]
        if self._snapshot_store is None:
            return ToolResult(error="snapshot history not enabled for this workspace")
        if is_internal_snapshot_path(path):
            return ToolResult(error="path is reserved for internal snapshot storage")
        try:
            if action == "list":
                entries = self._snapshot_store.history(path, params.get("limit", 20))
                return ToolResult(
                    output="\n".join(f"{entry.snapshot_id}\t{entry.timestamp}" for entry in entries) or "(no snapshots)",
                    metadata={"snapshots": [entry.snapshot_id for entry in entries]},
                )
            if action == "restore":
                snapshot_id = params.get("snapshot_id")
                if not isinstance(snapshot_id, str):
                    return ToolResult(error="restore requires snapshot_id")
                content = self._snapshot_store.restore(path, snapshot_id)
                try:
                    self._write_locked(path, lambda: self._workspace.write_text(path, content))
                except FileLockTimeoutError as exc:
                    return ToolResult(error=f"file is locked for writing: {exc}")
                return ToolResult(output=f"restored {path} from snapshot {snapshot_id}", metadata={"path": path, "snapshot_id": snapshot_id})
            return ToolResult(error="unknown action (expected list/restore)")
        except (SnapshotStoreError, WorkspaceError, ValueError, OSError) as exc:
            return ToolResult(error=f"snapshot history failed: {exc}")
