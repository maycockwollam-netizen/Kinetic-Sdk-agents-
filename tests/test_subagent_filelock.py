"""Tests for process-safe file coordination between sub-agents."""

from __future__ import annotations

import time

import pytest

from kinetic_sdk.subagent import (
    FileLockOwnershipError,
    FileLockRegistry,
    FileLockTimeoutError,
)
from kinetic_sdk.workspace import PathTraversalError


def test_acquire_release_and_context_manager(tmp_path):
    registry = FileLockRegistry(tmp_path)

    with registry.acquire("src/main.py", "agent-a") as lease:
        assert lease.path == str(tmp_path / "src" / "main.py")
        assert registry.owner_of("src/main.py").owner_id == "agent-a"  # type: ignore[union-attr]

    assert registry.owner_of("src/main.py") is None
    assert registry.locked_paths() == []


def test_conflicting_owner_times_out_and_same_owner_is_reentrant(tmp_path):
    registry = FileLockRegistry(tmp_path)
    registry.acquire("README.md", "agent-a", lease_seconds=None)
    second = registry.acquire("README.md", "agent-a")

    with pytest.raises(FileLockTimeoutError, match="agent-a"):
        registry.acquire("README.md", "agent-b", timeout=0)

    # One release ends a re-entrant ownership claim; this prevents hidden,
    # process-local ref-count state from leaving a cross-process stale lock.
    second.release()
    assert registry.owner_of("README.md") is None
    # Releasing an already-released lease itself remains a no-op.
    second.release()


def test_wrong_owner_cannot_release_or_renew(tmp_path):
    registry = FileLockRegistry(tmp_path)
    registry.acquire("notes.txt", "owner")

    with pytest.raises(FileLockOwnershipError, match="not 'intruder'"):
        registry.release("notes.txt", "intruder")
    with pytest.raises(FileLockOwnershipError, match="does not own"):
        registry.renew("notes.txt", "intruder")


def test_expired_lock_is_reclaimed_and_renewal_extends_it(tmp_path):
    registry = FileLockRegistry(tmp_path, default_lease_seconds=0.02)
    registry.acquire("generated.py", "old-owner")
    time.sleep(0.03)

    lease = registry.acquire("generated.py", "new-owner", timeout=0)
    renewed = registry.renew("generated.py", "new-owner", lease_seconds=1)
    assert renewed.owner_id == "new-owner"
    assert renewed.expires_at is not None and renewed.expires_at > time.time()
    lease.release()


def test_registry_is_shared_by_instances_and_release_all(tmp_path):
    first = FileLockRegistry(tmp_path)
    second = FileLockRegistry(tmp_path)
    first.acquire("one.py", "agent-a")
    first.acquire("two.py", "agent-a")

    assert [item.path for item in second.locked_paths()] == [
        str(tmp_path / "one.py"),
        str(tmp_path / "two.py"),
    ]
    assert second.release_all("agent-a") == 2
    assert first.locked_paths() == []


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py"])
def test_registry_rejects_paths_outside_workspace(tmp_path, path):
    registry = FileLockRegistry(tmp_path)
    with pytest.raises(PathTraversalError):
        registry.acquire(path, "agent-a", timeout=0)
