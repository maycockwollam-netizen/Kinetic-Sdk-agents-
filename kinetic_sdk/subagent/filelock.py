"""Process-safe file ownership registry for cooperating sub-agents.

The registry uses atomically-created JSON lock files rather than only an
in-memory mutex, so independently created agents (and even separate Python
processes) coordinating the same workspace see the same ownership.  Lock
files are deliberately stored outside the files being edited: locking a path
that does not exist yet is therefore supported.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from kinetic_sdk.subagent.exceptions import (
    FileLockOwnershipError,
    FileLockTimeoutError,
)
from kinetic_sdk.workspace.manager import LocalWorkspace, PathTraversalError

DEFAULT_LOCK_LEASE_SECONDS = 300.0
_POLL_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class FileLockInfo:
    """Public, non-secret description of a held file lock."""

    path: str
    owner_id: str
    acquired_at: float
    expires_at: float | None


class FileLock:
    """A held lock lease returned by :meth:`FileLockRegistry.acquire`.

    It is a context manager.  Calling :meth:`release` more than once is safe;
    an ownership mismatch, however, is never silently ignored.
    """

    def __init__(self, registry: FileLockRegistry, path: str, owner_id: str) -> None:
        self._registry = registry
        self.path = path
        self.owner_id = owner_id
        self._released = False

    def release(self) -> None:
        """Release this lease, if it has not already been released."""
        if not self._released:
            self._registry.release(self.path, self.owner_id)
            self._released = True

    def __enter__(self) -> FileLock:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class FileLockRegistry:
    """Coordinate exclusive ownership of workspace-relative files.

    Args:
        root_path: Workspace root.  Every requested path is resolved through
            :class:`LocalWorkspace`, rejecting traversal and symlink escapes.
        lock_dir: Optional directory for registry records.  The default is
            ``<root>/.kinetic/file-locks``.  It may be outside the workspace
            when callers want lock bookkeeping kept elsewhere.
        default_lease_seconds: Lifetime applied when ``lease_seconds`` is not
            supplied.  ``None`` disables expiry; positive values are required
            otherwise.  Expired records are reclaimed by the next acquirer.

    The owner id is supplied by the caller rather than inferred from a thread:
    this makes ownership stable across an agent's tool calls and auditable.
    Re-entrant acquisition by the same owner succeeds and returns a new lease;
    ownership is still released once (there is intentionally no hidden
    reference count spanning separate processes).
    """

    def __init__(
        self,
        root_path: str | os.PathLike[str],
        *,
        lock_dir: str | os.PathLike[str] | None = None,
        default_lease_seconds: float | None = DEFAULT_LOCK_LEASE_SECONDS,
    ) -> None:
        if default_lease_seconds is not None and default_lease_seconds <= 0:
            raise ValueError("default_lease_seconds must be positive or None")
        self._workspace = LocalWorkspace(root_path)
        directory = (
            Path(lock_dir)
            if lock_dir is not None
            else Path(self._workspace.root_path) / ".kinetic" / "file-locks"
        )
        self._lock_dir = directory.resolve()
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        if not self._lock_dir.is_dir():
            raise ValueError(f"lock_dir is not a directory: {self._lock_dir}")
        self.default_lease_seconds = default_lease_seconds
        self._mutex = threading.RLock()

    @property
    def root_path(self) -> str:
        """Canonical workspace root whose paths this registry protects."""
        return self._workspace.root_path

    @property
    def lock_dir(self) -> str:
        """Canonical directory containing opaque registry records."""
        return str(self._lock_dir)

    def acquire(
        self,
        path: str,
        owner_id: str,
        *,
        timeout: float | None = None,
        lease_seconds: float | None = None,
    ) -> FileLock:
        """Acquire exclusive ownership of *path* and return a lease.

        ``timeout=None`` waits indefinitely; ``timeout=0`` performs one
        non-blocking attempt.  A conflicting, unexpired owner raises
        :class:`FileLockTimeoutError` after the timeout.  Stale locks are
        automatically removed before retrying.
        """
        canonical = self._canonical_path(path)
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        if lease_seconds is None:
            lease_seconds = self.default_lease_seconds
        if lease_seconds is not None and lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            now = time.time()
            info = FileLockInfo(
                path=canonical,
                owner_id=owner_id,
                acquired_at=now,
                expires_at=None if lease_seconds is None else now + lease_seconds,
            )
            if self._try_create(info):
                return FileLock(self, canonical, owner_id)
            existing = self._read_info(canonical)
            if existing is not None and existing.owner_id == owner_id:
                return FileLock(self, canonical, owner_id)
            if existing is not None and self._is_expired(existing, now):
                self._remove_if_unchanged(canonical, existing)
                continue
            if deadline is not None and time.monotonic() >= deadline:
                holder = existing.owner_id if existing is not None else "unknown"
                raise FileLockTimeoutError(
                    f"Timed out acquiring file lock for {path!r}; held by {holder!r}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

    def release(self, path: str, owner_id: str) -> None:
        """Release *path* only when it is currently owned by *owner_id*."""
        canonical = self._canonical_path(path)
        with self._mutex:
            info = self._read_info(canonical)
            if info is None:
                raise FileLockOwnershipError(f"No file lock is held for {path!r}")
            if info.owner_id != owner_id:
                raise FileLockOwnershipError(
                    f"File lock for {path!r} is owned by {info.owner_id!r}, not {owner_id!r}"
                )
            try:
                self._record_path(canonical).unlink()
            except FileNotFoundError:
                raise FileLockOwnershipError(f"No file lock is held for {path!r}") from None

    def renew(self, path: str, owner_id: str, *, lease_seconds: float | None = None) -> FileLockInfo:
        """Extend a lock lease while preserving its original acquisition time."""
        canonical = self._canonical_path(path)
        if lease_seconds is None:
            lease_seconds = self.default_lease_seconds
        if lease_seconds is not None and lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive or None")
        with self._mutex:
            info = self._read_info(canonical)
            if info is None or info.owner_id != owner_id:
                raise FileLockOwnershipError(f"Agent {owner_id!r} does not own {path!r}")
            updated = FileLockInfo(
                canonical, owner_id, info.acquired_at,
                None if lease_seconds is None else time.time() + lease_seconds,
            )
            self._write_record(canonical, updated, replace=True)
            return updated

    def owner_of(self, path: str) -> FileLockInfo | None:
        """Return current ownership, reclaiming an expired record first."""
        canonical = self._canonical_path(path)
        info = self._read_info(canonical)
        if info is not None and self._is_expired(info, time.time()):
            self._remove_if_unchanged(canonical, info)
            return None
        return info

    def locked_paths(self) -> list[FileLockInfo]:
        """List all live locks, sorted by canonical protected path."""
        live: list[FileLockInfo] = []
        for record in self._lock_dir.glob("*.lock"):
            info = self._read_record(record)
            if info is None:
                continue
            if self._is_expired(info, time.time()):
                self._remove_if_unchanged(info.path, info)
            else:
                live.append(info)
        return sorted(live, key=lambda item: item.path)

    def release_all(self, owner_id: str) -> int:
        """Release every currently-held lock of an agent; return its count."""
        released = 0
        for info in self.locked_paths():
            if info.owner_id == owner_id:
                try:
                    self.release(info.path, owner_id)
                except FileLockOwnershipError:
                    continue
                released += 1
        return released

    def _canonical_path(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        try:
            return self._workspace.resolve(path)
        except PathTraversalError:
            raise

    def _record_path(self, canonical: str) -> Path:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._lock_dir / f"{digest}.lock"

    def _try_create(self, info: FileLockInfo) -> bool:
        record = self._record_path(info.path)
        payload = self._payload(info)
        try:
            fd = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            record.unlink(missing_ok=True)
            raise
        return True

    def _write_record(self, canonical: str, info: FileLockInfo, *, replace: bool) -> None:
        record = self._record_path(canonical)
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._payload(info), sort_keys=True), encoding="utf-8")
        if replace:
            os.replace(temporary, record)
        else:  # pragma: no cover - reserved for future atomic update paths
            os.link(temporary, record)
            temporary.unlink()

    @staticmethod
    def _payload(info: FileLockInfo) -> dict[str, str | float | None]:
        return {
            "path": info.path,
            "owner_id": info.owner_id,
            "acquired_at": info.acquired_at,
            "expires_at": info.expires_at,
        }

    def _read_info(self, canonical: str) -> FileLockInfo | None:
        return self._read_record(self._record_path(canonical))

    @staticmethod
    def _read_record(record: Path) -> FileLockInfo | None:
        try:
            raw = json.loads(record.read_text(encoding="utf-8"))
            path, owner_id = raw["path"], raw["owner_id"]
            acquired_at, expires_at = raw["acquired_at"], raw["expires_at"]
            if not isinstance(path, str) or not isinstance(owner_id, str):
                return None
            if not isinstance(acquired_at, (int, float)) or not isinstance(expires_at, (int, float, type(None))):
                return None
            return FileLockInfo(path, owner_id, float(acquired_at), None if expires_at is None else float(expires_at))
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _is_expired(info: FileLockInfo, now: float) -> bool:
        return info.expires_at is not None and info.expires_at <= now

    def _remove_if_unchanged(self, canonical: str, expected: FileLockInfo) -> None:
        with self._mutex:
            current = self._read_info(canonical)
            if current == expected:
                self._record_path(canonical).unlink(missing_ok=True)
