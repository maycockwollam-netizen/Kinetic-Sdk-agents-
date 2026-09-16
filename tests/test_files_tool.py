"""File locking behaviour wired into :class:`kinetic_sdk.files.FileTool`."""

from __future__ import annotations

from kinetic_sdk.files import FileTool
from kinetic_sdk.subagent import FileLockRegistry, agent_id_for
from kinetic_sdk.workspace import LocalWorkspace


class _BoundAgent:
    """Weak-referenceable stand-in used solely to obtain delegation ids."""


class _FailingWriteWorkspace(LocalWorkspace):
    def write_text(self, relative_path: str, content: str) -> None:
        raise OSError("simulated write failure")


def test_shared_registry_rejects_conflicting_filetool_writer(tmp_path):
    workspace = LocalWorkspace(tmp_path)
    registry = FileLockRegistry(tmp_path)
    first_agent = _BoundAgent()
    second_agent = _BoundAgent()
    first = FileTool(workspace, lock_registry=registry, lock_timeout=0)
    second = FileTool(workspace, lock_registry=registry, lock_timeout=0)
    first.bind(first_agent)  # distinct owner ids come from delegation audit ids
    second.bind(second_agent)

    with registry.acquire("shared.txt", agent_id_for(first_agent)):
        result = second.execute(action="create", path="shared.txt", file_text="second")

    assert result.error is not None
    assert "shared.txt" in result.error
    assert agent_id_for(first_agent) in result.error
    assert not (tmp_path / "shared.txt").exists()


def test_view_is_not_blocked_by_another_agents_write_lock(tmp_path):
    (tmp_path / "shared.txt").write_text("visible\n", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path)
    registry = FileLockRegistry(tmp_path)
    holder = _BoundAgent()
    reader = FileTool(workspace, lock_registry=registry, lock_timeout=0)
    reader.bind(_BoundAgent())

    with registry.acquire("shared.txt", agent_id_for(holder)):
        result = reader.execute(action="view", path="shared.txt")

    assert result.error is None
    assert result.output == "1\tvisible"


def test_filetool_without_registry_keeps_existing_write_behaviour(tmp_path):
    tool = FileTool(LocalWorkspace(tmp_path))

    result = tool.execute(action="create", path="plain.txt", file_text="works")

    assert result.error is None
    assert (tmp_path / "plain.txt").read_text(encoding="utf-8") == "works"


def test_write_lock_releases_after_success_and_backend_failure(tmp_path):
    registry = FileLockRegistry(tmp_path)
    successful = FileTool(LocalWorkspace(tmp_path), lock_registry=registry)
    successful_agent = _BoundAgent()
    successful.bind(successful_agent)

    success = successful.execute(action="create", path="success.txt", file_text="ok")

    assert success.error is None
    assert registry.owner_of("success.txt") is None

    (tmp_path / "failure.txt").write_text("before", encoding="utf-8")
    failing = FileTool(_FailingWriteWorkspace(tmp_path), lock_registry=registry)
    failing_agent = _BoundAgent()
    failing.bind(failing_agent)

    failure = failing.execute(
        action="str_replace", path="failure.txt", old_str="before", new_str="after"
    )

    assert failure.error == "filesystem error: simulated write failure"
    assert registry.owner_of("failure.txt") is None
