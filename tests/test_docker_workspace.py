"""Tests for ``DockerWorkspace`` path handling and command building.

No Docker daemon is required: ``run_command`` is replaced per-instance with a
recording stand-in, or the traversal checks are exercised before any
subprocess could be spawned.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.workspace import DockerWorkspace
from kinetic_sdk.workspace.base import CommandResult, WorkspaceError
from kinetic_sdk.workspace.manager import PathTraversalError


class _RecordingRunner:
    """Instance-level stand-in for ``run_command`` that records calls."""

    def __init__(self, *, exit_code: int = 0, output: str = "") -> None:
        self.calls: list[tuple[str, dict]] = []
        self._exit_code = exit_code
        self._output = output

    def __call__(self, command: str, **kwargs: object) -> CommandResult:
        self.calls.append((command, kwargs))
        return CommandResult(
            output=self._output, exit_code=self._exit_code, duration_seconds=0.0
        )


def _workspace(**kwargs: object) -> DockerWorkspace:
    defaults: dict[str, object] = {"mode": "exec", "container": "dev"}
    defaults.update(kwargs)
    return DockerWorkspace(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "method,args",
    [
        ("read_text", ("../../etc/passwd",)),
        ("write_text", ("../../tmp/evil.txt", "x")),
        ("delete_file", ("../../tmp/evil.txt",)),
        ("list_directory", ("../../etc",)),
    ],
)
def test_docker_workspace_rejects_traversal(method: str, args: tuple[object, ...]) -> None:
    ws = _workspace()
    ws.run_command = _RecordingRunner()  # type: ignore[method-assign]
    with pytest.raises(PathTraversalError):
        getattr(ws, method)(*args)


@pytest.mark.parametrize("relative", ["foo/../../etc", "/etc/passwd", "a/../../../etc"])
def test_docker_workspace_rejects_absolute_and_mid_traversal(relative: str) -> None:
    ws = _workspace()
    with pytest.raises(PathTraversalError):
        ws._absolute_path(relative)  # noqa: SLF001


def test_docker_workspace_contained_paths_are_accepted() -> None:
    ws = _workspace()
    assert ws._absolute_path("a.txt") == "/workspace/a.txt"  # noqa: SLF001
    assert ws._absolute_path("sub/dir/f.txt") == "/workspace/sub/dir/f.txt"  # noqa: SLF001
    assert ws._absolute_path("./a.txt") == "/workspace/a.txt"  # noqa: SLF001
    assert ws._absolute_path("sub/../a.txt") == "/workspace/a.txt"  # noqa: SLF001


def test_docker_workspace_read_text_builds_cat_inside_root() -> None:
    ws = _workspace()
    runner = _RecordingRunner(output="file content")
    ws.run_command = runner  # type: ignore[method-assign]
    assert ws.read_text("a.txt") == "file content"
    assert runner.calls == [("cat /workspace/a.txt", {})]


def test_docker_workspace_write_text_pipes_content_via_stdin() -> None:
    ws = _workspace()
    runner = _RecordingRunner()
    ws.run_command = runner  # type: ignore[method-assign]
    ws.write_text("sub/dir.txt", "hello\nworld")
    assert [command for command, _ in runner.calls] == [
        "mkdir -p /workspace/sub",
        "cat > /workspace/sub/dir.txt",
    ]
    assert runner.calls[1][1] == {"input_text": "hello\nworld"}


def test_docker_workspace_delete_file_builds_rm() -> None:
    ws = _workspace()
    runner = _RecordingRunner()
    ws.run_command = runner  # type: ignore[method-assign]
    ws.delete_file("a.txt")
    assert runner.calls[0][0] == "rm -- /workspace/a.txt"


def test_docker_workspace_list_files_and_directory_within_root() -> None:
    ws = _workspace()
    runner = _RecordingRunner(output="./a.py\n./dir/b.py\n./c.txt")
    ws.run_command = runner  # type: ignore[method-assign]
    assert ws.list_files("*.py") == ["a.py", "dir/b.py"]

    runner = _RecordingRunner(output="a.txtf\nbd\n")
    ws.run_command = runner  # type: ignore[method-assign]
    assert ws.list_directory(".") == ["a.txt", "b/"]


def test_docker_workspace_rejects_cwd_traversal_before_spawning() -> None:
    ws = _workspace()
    with pytest.raises(PathTraversalError):
        ws.run_command("ls", cwd="../../etc")


def test_docker_workspace_run_mode_refuses_file_operations() -> None:
    ws = DockerWorkspace(mode="run", image="python:3.12")
    with pytest.raises(WorkspaceError):
        ws.read_text("a.txt")
    with pytest.raises(WorkspaceError):
        ws.write_text("a.txt", "x")
    with pytest.raises(WorkspaceError):
        ws.delete_file("a.txt")


def test_docker_workspace_constructor_validation() -> None:
    with pytest.raises(ValueError, match="container"):
        DockerWorkspace(mode="exec")
    with pytest.raises(ValueError, match="image"):
        DockerWorkspace(mode="run")
    with pytest.raises(ValueError, match="absolute"):
        DockerWorkspace(mode="exec", container="c", workdir="workspace")

    ws = DockerWorkspace(mode="exec", container="c", workdir="/workspace/")
    assert ws.root_path == "/workspace"


def test_docker_workspace_container_name() -> None:
    assert _workspace().container_name == "dev"
    run_ws = DockerWorkspace(mode="run", image="python:3.12")
    assert run_ws.container_name is None


def test_default_timeout_validated() -> None:
    with pytest.raises(ValueError, match="default_timeout"):
        DockerWorkspace(mode="exec", container="c", default_timeout=-1)
    with pytest.raises(ValueError, match="default_timeout"):
        DockerWorkspace(mode="exec", container="c", default_timeout=float("nan"))


def test_run_command_applies_default_timeout(monkeypatch) -> None:
    import subprocess

    ws = _workspace()
    recorded: dict[str, object] = {}

    class FakeProc:
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def communicate(self, input: object = None, timeout: object = None):  # type: ignore[no-untyped-def]
            recorded["timeout"] = timeout
            return "ok", None

    monkeypatch.setattr(subprocess, "Popen", FakeProc)

    ws.run_command("cat /x")  # no explicit timeout -> default applies
    assert recorded["timeout"] == ws._default_timeout  # noqa: SLF001
    ws.run_command("cat /x", timeout=7)
    assert recorded["timeout"] == 7


def test_file_ops_leave_timeout_to_run_command_default() -> None:
    ws = _workspace(default_timeout=25)
    seen: list[dict] = []

    def record(command: str, **kwargs: object) -> CommandResult:
        seen.append(kwargs)
        return CommandResult(output="ok", exit_code=0, duration_seconds=0.0)

    ws.run_command = record  # type: ignore[method-assign]
    ws.read_text("a.txt")
    ws.write_text("a.txt", "x")
    ws.delete_file("a.txt")
    ws.list_files()
    ws.list_directory(".")
    assert seen
    # None of the file ops pass their own timeout; run_command's default is used.
    assert all("timeout" not in kwargs for kwargs in seen)


def test_read_text_timeout_reaches_the_subprocess(monkeypatch) -> None:
    import subprocess

    ws = _workspace(default_timeout=11)
    recorded: dict[str, object] = {}

    class FakeProc:
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def communicate(self, input: object = None, timeout: object = None):  # type: ignore[no-untyped-def]
            recorded["timeout"] = timeout
            return "file content", None

    monkeypatch.setattr(subprocess, "Popen", FakeProc)

    assert ws.read_text("a.txt") == "file content"
    assert recorded["timeout"] == 11
