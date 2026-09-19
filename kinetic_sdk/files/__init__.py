"""Files package: workspace-scoped viewing, editing, patching, search, and history tools."""

from kinetic_sdk.files.history import FileHistoryTool
from kinetic_sdk.files.patch import ApplyPatchTool, PatchApplyError
from kinetic_sdk.files.search import GlobTool, GrepTool
from kinetic_sdk.files.snapshots import (
    GitSnapshotStore,
    SnapshotEntry,
    SnapshotStoreError,
)
from kinetic_sdk.files.tool import FileTool

__all__ = [
    "ApplyPatchTool", "FileHistoryTool", "FileTool", "GitSnapshotStore",
    "GlobTool", "GrepTool", "PatchApplyError", "SnapshotEntry", "SnapshotStoreError",
]
