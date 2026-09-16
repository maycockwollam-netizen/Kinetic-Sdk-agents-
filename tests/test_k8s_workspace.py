"""Offline argv and error-contract tests for ``KubernetesWorkspace``."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from kinetic_sdk.workspace import (
    KubernetesWorkspace,
    PathTraversalError,
    WorkspaceError,
)


def _success(*_args: object, **_kwargs: object) -> object:
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_kubernetes_workspace_builds_exact_exec_argv() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="contents", stderr="")

    workspace = KubernetesWorkspace(
        pod_name="agent-0", namespace="coding", container="sidecar", runner=runner
    )
    assert workspace.run_command("echo hello", cwd="src").output == "contents"
    assert workspace.read_text("README.md") == "contents"
    workspace.write_text("nested/note.txt", "hello")

    prefix = ["kubectl", "-n", "coding", "exec", "agent-0", "-c", "sidecar", "--"]
    assert calls[0][0] == [*prefix, "bash", "-c", "cd /workspace/src && echo hello"]
    assert calls[1][0] == [*prefix, "cat", "/workspace/README.md"]
    assert calls[2][0] == [*prefix, "sh", "-c", "mkdir -p /workspace/nested && cat > /workspace/nested/note.txt"]
    assert calls[2][1]["input"] == "hello"
    assert calls[0][1]["timeout"] == 30.0


def test_kubernetes_workspace_timeout_becomes_workspace_error() -> None:
    def runner(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("kubectl", 30)

    workspace = KubernetesWorkspace(pod_name="agent-0", runner=runner)
    with pytest.raises(WorkspaceError, match="timed out after 30.0s"):
        workspace.run_command("echo never")


def test_kubernetes_workspace_requires_pod_name() -> None:
    with pytest.raises(ValueError, match="pod_name must be a non-empty string"):
        KubernetesWorkspace(pod_name="", runner=_success)


@pytest.mark.parametrize("path", ["../etc/passwd", "nested/../../etc/passwd", "/etc/passwd"])
def test_kubernetes_workspace_rejects_path_traversal(path: str) -> None:
    workspace = KubernetesWorkspace(pod_name="agent-0", runner=_success)
    with pytest.raises(PathTraversalError, match="outside the workspace root"):
        workspace.read_text(path)


def test_kubernetes_workspace_rejects_traversal_in_command_cwd() -> None:
    workspace = KubernetesWorkspace(pod_name="agent-0", runner=_success)
    with pytest.raises(PathTraversalError):
        workspace.run_command("echo hello", cwd="../etc")
