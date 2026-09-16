"""``KubernetesWorkspace``: operate in an existing Kubernetes pod.

The common platform deployment already provisions a coding-agent sidecar, so
this backend deliberately connects to that pod rather than creating Kubernetes
resources itself.  It uses the ``kubectl`` CLI (like ``DockerWorkspace`` uses
``docker``), keeping the SDK core stdlib-only and making cluster credentials,
contexts and RBAC the platform operator's concern.

Files are read and written through ``kubectl exec`` instead of ``kubectl cp``.
``cp`` requires ``tar`` in the target image and varies across minimal sidecars;
``cat`` over exec has fewer image-level assumptions and lets ``write_text``
create parent directories atomically in the same shell invocation.
"""

from __future__ import annotations

import fnmatch
import posixpath
import subprocess
import time
from shlex import quote
from typing import Callable, cast

from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase, WorkspaceError
from kinetic_sdk.workspace.manager import PathTraversalError


class KubernetesWorkspace(WorkspaceBase):
    """A persistent workspace backed by an already-running Kubernetes pod.

    Args:
        pod_name: Name of the existing pod to execute in.
        namespace: Optional Kubernetes namespace.  The current kubectl context
            namespace is used when omitted.
        container: Optional container name for multi-container pods.
        workdir: Root directory inside the pod.
        timeout: Default timeout applied to every ``kubectl`` invocation.
        runner: Injectable ``subprocess.run``-compatible callable, which keeps
            unit tests independent from a cluster and kubectl installation.
    """

    def __init__(
        self,
        *,
        pod_name: str,
        namespace: str | None = None,
        container: str | None = None,
        workdir: str = "/workspace",
        timeout: float = 30.0,
        runner: Callable[..., object] = subprocess.run,
    ) -> None:
        if not pod_name or not pod_name.strip():
            raise ValueError("pod_name must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._pod_name = pod_name
        self._namespace = namespace
        self._container = container
        if not workdir.startswith("/"):
            raise ValueError("workdir must be an absolute POSIX path")
        self._workdir = posixpath.normpath(workdir)
        self._timeout = timeout
        self._runner = runner

    @property
    def root_path(self) -> str:
        return self._workdir

    def _exec_argv(self, *command: str) -> list[str]:
        argv = ["kubectl"]
        if self._namespace:
            argv.extend(["-n", self._namespace])
        argv.extend(["exec", self._pod_name])
        if self._container:
            argv.extend(["-c", self._container])
        return [*argv, "--", *command]

    def _run_kubectl(
        self, argv: list[str], *, input_text: str | None = None, timeout: float | None = None
    ) -> object:
        effective_timeout = self._timeout if timeout is None else timeout
        try:
            return self._runner(
                argv,
                input=input_text,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceError(f"kubectl command timed out after {effective_timeout}s") from exc
        except OSError as exc:
            raise WorkspaceError(f"failed to start kubectl: {exc}") from exc

    @staticmethod
    def _output(proc: object) -> str:
        stdout = cast(str | None, getattr(proc, "stdout", "")) or ""
        stderr = cast(str | None, getattr(proc, "stderr", "")) or ""
        return stdout + stderr

    def _absolute_path(self, relative_path: str) -> str:
        candidate = posixpath.normpath(
            relative_path
            if posixpath.isabs(relative_path)
            else posixpath.join(self._workdir, relative_path)
        )
        try:
            inside = posixpath.commonpath([self._workdir, candidate]) == self._workdir
        except ValueError:
            inside = False
        if not inside:
            raise PathTraversalError(
                f"path {relative_path!r} resolves outside the workspace root {self._workdir!r}"
            )
        return candidate

    def run_command(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None
    ) -> CommandResult:
        """Execute shell text through ``kubectl exec`` in the selected pod."""
        effective = command if cwd is None else f"cd {quote(self._absolute_path(cwd))} && {command}"
        started = time.monotonic()
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout must be positive")
        proc = self._run_kubectl(self._exec_argv("bash", "-c", effective), timeout=timeout)
        return CommandResult(
            output=self._output(proc),
            exit_code=cast(int, getattr(proc, "returncode", -1)),
            duration_seconds=round(time.monotonic() - started, 3),
        )

    def read_text(self, relative_path: str) -> str:
        proc = self._run_kubectl(self._exec_argv("cat", self._absolute_path(relative_path)))
        output = self._output(proc)
        if getattr(proc, "returncode", -1) != 0:
            raise WorkspaceError(f"failed to read {relative_path!r} from pod: {output}")
        return cast(str, getattr(proc, "stdout", ""))

    def write_text(self, relative_path: str, content: str) -> None:
        absolute = self._absolute_path(relative_path)
        parent = absolute.rsplit("/", 1)[0] or "/"
        proc = self._run_kubectl(
            self._exec_argv("sh", "-c", f"mkdir -p {quote(parent)} && cat > {quote(absolute)}"),
            input_text=content,
        )
        if getattr(proc, "returncode", -1) != 0:
            raise WorkspaceError(f"failed to write {relative_path!r}: {self._output(proc)}")

    def delete_file(self, relative_path: str) -> None:
        proc = self._run_kubectl(self._exec_argv("rm", "--", self._absolute_path(relative_path)))
        if getattr(proc, "returncode", -1) != 0:
            raise WorkspaceError(f"failed to delete {relative_path!r}: {self._output(proc)}")

    def list_files(self, pattern: str | None = None) -> list[str]:
        proc = self._run_kubectl(
            self._exec_argv("sh", "-c", f"cd {quote(self._workdir)} && find . -type f")
        )
        output = self._output(proc)
        if getattr(proc, "returncode", -1) != 0:
            raise WorkspaceError(f"failed to list files: {output}")
        files = [line[2:] for line in cast(str, getattr(proc, "stdout", "")).splitlines() if line.startswith("./")]
        if pattern is not None:
            files = [file_path for file_path in files if fnmatch.fnmatch(file_path, pattern)]
        return sorted(files)

    def list_directory(self, relative_path: str = ".") -> list[str]:
        directory = self._absolute_path(relative_path)
        proc = self._run_kubectl(
            self._exec_argv("find", directory, "-mindepth", "1", "-maxdepth", "1", "-printf", "%f%y\\n")
        )
        output = self._output(proc)
        if getattr(proc, "returncode", -1) != 0:
            raise WorkspaceError(f"failed to list directory: {output}")
        return sorted(line[:-1] + "/" if line.endswith("d") else line[:-1] for line in cast(str, getattr(proc, "stdout", "")).splitlines())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<KubernetesWorkspace pod={self._pod_name!r} workdir={self._workdir!r}>"
