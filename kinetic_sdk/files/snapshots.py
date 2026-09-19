"""Durable, private git-backed snapshots for workspace file edits."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from kinetic_sdk.git import GitRunner
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

try:  # ``fcntl`` is unavailable on Windows, where the in-process lock remains.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]


class SnapshotStoreError(RuntimeError):
    """Raised when the private snapshot repository cannot serve a request."""


@dataclass(frozen=True)
class SnapshotEntry:
    """One durable version of a workspace-relative file."""

    snapshot_id: str
    path: str
    timestamp: int


SNAPSHOT_DIRECTORY = ".kinetic/snapshots"
DEFAULT_EXCLUDE_GLOBS = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "*.p12", "credentials*", "secrets*")
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{7,64}$")
_LOG = logging.getLogger(__name__)
_LOCKS_GUARD = threading.Lock()
_REPOSITORY_LOCKS: dict[str, threading.RLock] = {}
_EXCLUSION_WARNED: set[tuple[str, str]] = set()


def is_internal_snapshot_path(path: str) -> bool:
    """Whether *path* addresses the SDK's private snapshot area."""
    return ".kinetic" in Path(path).parts


class GitSnapshotStore:
    """Private, isolated git storage for pre-edit workspace file contents.

    The sidecar repository deliberately ignores both user git configuration and
    hooks.  It is also hidden from the user's repository by a nested
    ``.kinetic/.gitignore``, rather than by modifying the user's ``.gitignore``.
    """

    def __init__(
        self,
        workspace: WorkspaceBase,
        runner: GitRunner | None = None,
        *,
        exclude_globs: tuple[str, ...] = DEFAULT_EXCLUDE_GLOBS,
        strict: bool = True,
    ) -> None:
        self._workspace = workspace
        try:
            root = Path(workspace.resolve("."))
        except WorkspaceError as exc:
            raise ValueError("GitSnapshotStore requires a local resolving workspace") from exc
        self._root = root
        self._repo_dir = root / SNAPSHOT_DIRECTORY
        self._runner: GitRunner = runner if runner is not None else self._default_git_runner
        self._exclude_globs = tuple(exclude_globs)
        self._strict = strict
        self._lock = self._lock_for_repo(self._repo_dir)

    def snapshot(self, path: str, content: str) -> str:
        """Commit *content* as a durable version of workspace-relative *path*.

        Excluded paths return an empty id without ever writing their content.
        In non-strict mode git failures are fail-open so a file edit can proceed.
        """
        relative = self._validate_path(path)
        if self._is_excluded(relative):
            self._warn_excluded_once(relative)
            return ""
        with self._snapshot_lock():
            try:
                self._ensure_repo()
                target = self._repo_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                self._run(["add", "--", relative.as_posix()])
                self._run([
                    "-c", "user.name=Kinetic Snapshot Store",
                    "-c", "user.email=snapshots@kinetic.invalid",
                    "commit", "--no-verify", "--quiet", "-m", f"Snapshot {relative.as_posix()}", "--", relative.as_posix(),
                ])
                return self._run(["rev-parse", "HEAD"]).stdout.strip()
            except SnapshotStoreError:
                if self._strict:
                    raise
                _LOG.warning("snapshot failed for %s; continuing because strict=False", relative)
                return ""

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

    @staticmethod
    def _lock_for_repo(repo_dir: Path) -> threading.RLock:
        key = os.path.realpath(repo_dir)
        with _LOCKS_GUARD:
            return _REPOSITORY_LOCKS.setdefault(key, threading.RLock())

    @contextmanager
    def _snapshot_lock(self) -> Iterator[None]:
        """Serialize snapshots across threads and, where supported, processes."""
        with self._lock:
            lock_path = self._repo_dir.parent / "snapshots.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_repo(self) -> None:
        if (self._repo_dir / ".git").is_dir():
            return
        self._repo_dir.mkdir(parents=True, exist_ok=True)
        # A nested ignore file hides the whole sidecar directory without
        # changing the user's repository or its .gitignore.
        ignore_file = self._repo_dir.parent / ".gitignore"
        if not ignore_file.exists():
            ignore_file.write_text("*\n")
        self._run(["init", "--quiet"])

    def _ensure_exists(self) -> None:
        if not (self._repo_dir / ".git").is_dir():
            raise SnapshotStoreError("snapshot history not enabled for this workspace")

    def _is_excluded(self, relative: Path) -> bool:
        value = relative.as_posix()
        return any(fnmatch.fnmatch(value, glob) or fnmatch.fnmatch(relative.name, glob) for glob in self._exclude_globs)

    def _warn_excluded_once(self, relative: Path) -> None:
        key = (os.path.realpath(self._repo_dir), relative.as_posix())
        with _LOCKS_GUARD:
            if key in _EXCLUSION_WARNED:
                return
            _EXCLUSION_WARNED.add(key)
        _LOG.warning("not snapshotting excluded path %s", relative)

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
            return resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("path resolves outside the workspace") from exc

    @staticmethod
    def _validate_snapshot_id(snapshot_id: str) -> None:
        if not isinstance(snapshot_id, str) or not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise ValueError("snapshot_id must be a git commit hash")

    @staticmethod
    def _default_git_runner(argv: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        # Git has several environment forms for injecting configuration and
        # repository paths (including GIT_CONFIG_COUNT/KEY/VALUE).  Discard all
        # of them before applying the sidecar's explicitly safe configuration.
        for name in tuple(env):
            if name.startswith("GIT_"):
                env.pop(name)
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        isolated_args = [
            "git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args,
        ]
        try:
            completed = self._runner(isolated_args, str(self._repo_dir), 30.0)
        except FileNotFoundError as exc:
            raise SnapshotStoreError("git executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SnapshotStoreError("snapshot git command timed out") from exc
        if completed.returncode != 0:
            raise SnapshotStoreError((completed.stderr or completed.stdout or "snapshot git command failed").strip())
        return completed
