"""Durable, private git-backed snapshots for workspace file edits."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kinetic_sdk.git import GitRunner
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError


class SnapshotStoreError(RuntimeError):
    """Raised when the private snapshot repository cannot serve a request."""


@dataclass(frozen=True)
class SnapshotEntry:
    """One durable version of a workspace-relative file."""

    snapshot_id: str
    path: str
    timestamp: int


SNAPSHOT_DIRECTORY = ".kinetic/snapshots"
_IGNORE_ENTRY = "/.kinetic/snapshots/"
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{7,64}$")


def is_internal_snapshot_path(path: str) -> bool:
    """Whether *path* addresses the SDK's private snapshot area."""
    return ".kinetic" in Path(path).parts


class GitSnapshotStore:
    """Not ``GitTool``: a private sidecar git store, never the user's repository.

    The repository lives at ``<workspace>/.kinetic/snapshots`` and is created
    lazily. It is an entirely separate repository, used solely as durable
    storage for pre-edit file contents; it never runs git in, commits to, or
    otherwise modifies the user's real repository, branches, or worktree.
    """

    def __init__(self, workspace: WorkspaceBase, runner: GitRunner | None = None) -> None:
        self._workspace = workspace
        try:
            root = Path(workspace.resolve("."))
        except WorkspaceError as exc:
            raise ValueError("GitSnapshotStore requires a local resolving workspace") from exc
        self._root = root
        self._repo_dir = root / SNAPSHOT_DIRECTORY
        self._runner: GitRunner = runner if runner is not None else self._default_git_runner

    def snapshot(self, path: str, content: str) -> str:
        """Commit *content* as a durable version of workspace-relative *path*."""
        relative = self._validate_path(path)
        self._ensure_repo()
        target = self._repo_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self._run(["add", "--", relative.as_posix()])
        self._run([
            "-c", "user.name=Kinetic Snapshot Store",
            "-c", "user.email=snapshots@kinetic.invalid",
            "commit", "--quiet", "-m", f"Snapshot {relative.as_posix()}", "--", relative.as_posix(),
        ])
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def restore(self, path: str, snapshot_id: str) -> str:
        """Read *path* at *snapshot_id* without changing the workspace file."""
        relative = self._validate_path(path)
        self._validate_snapshot_id(snapshot_id)
        self._ensure_exists()
        return self._run(["show", f"{snapshot_id}:{relative.as_posix()}"]).stdout

    def history(self, path: str, limit: int = 20) -> list[SnapshotEntry]:
        """Return up to *limit* snapshots of *path*, newest first."""
        relative = self._validate_path(path)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not self._repo_dir.is_dir():
            return []
        output = self._run([
            "log", f"--max-count={limit}", "--format=%H%x00%ct", "--", relative.as_posix(),
        ]).stdout
        entries: list[SnapshotEntry] = []
        for line in output.splitlines():
            snapshot_id, separator, timestamp = line.partition("\0")
            if separator and timestamp.isdigit():
                entries.append(SnapshotEntry(snapshot_id, relative.as_posix(), int(timestamp)))
        return entries

    def _ensure_repo(self) -> None:
        if (self._repo_dir / ".git").is_dir():
            return
        self._repo_dir.mkdir(parents=True, exist_ok=True)
        self._run(["init", "--quiet"])
        ignore_file = self._root / ".gitignore"
        if ignore_file.exists():
            existing = ignore_file.read_text()
            if _IGNORE_ENTRY not in existing.splitlines():
                with ignore_file.open("a") as handle:
                    if existing and not existing.endswith("\n"):
                        handle.write("\n")
                    handle.write(f"{_IGNORE_ENTRY}\n")

    def _ensure_exists(self) -> None:
        if not (self._repo_dir / ".git").is_dir():
            raise SnapshotStoreError("snapshot history not enabled for this workspace")

    def _validate_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty workspace-relative string")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts or is_internal_snapshot_path(path):
            raise ValueError("path must not address internal snapshot storage or escape the workspace")
        try:
            resolved = Path(self._workspace.resolve(path))
        except WorkspaceError as exc:
            raise ValueError(f"path is not available in this workspace: {path!r}") from exc
        try:
            relative = resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("path resolves outside the workspace") from exc
        return relative

    @staticmethod
    def _validate_snapshot_id(snapshot_id: str) -> None:
        if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise ValueError("snapshot_id must be a git commit hash")

    @staticmethod
    def _default_git_runner(argv: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._runner(["git", *args], str(self._repo_dir), 30.0)
        except FileNotFoundError as exc:
            raise SnapshotStoreError("git executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SnapshotStoreError("snapshot git command timed out") from exc
        if completed.returncode != 0:
            raise SnapshotStoreError((completed.stderr or completed.stdout or "snapshot git command failed").strip())
        return completed
