"""``DockerWorkspace``: run/read/write inside a Docker container.

Two modes, matching the two wrappers already in ``terminal/docker.py``:

* ``exec`` mode (persistent container you started yourself) — supports
  ``read_text``/``write_text`` by shelling out (``cat`` / ``sh -c 'echo ... >
  file'``) since the same container is reused across calls.
* ``run`` mode (fresh ``--rm`` container per command) — ``read_text`` and
  ``write_text`` raise :class:`~kinetic_sdk.workspace.base.WorkspaceError`,
  since there is no persistent filesystem between calls to read back from or
  write into ahead of time.

No new subprocess logic is written here: both modes build an argv prefix
with the existing wrapper functions, then run it exactly the way
:class:`~kinetic_sdk.terminal.tool.TerminalTool` already runs
``command_wrapper``-prefixed commands.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from shlex import quote

from kinetic_sdk.terminal.docker import docker_exec_wrapper, docker_run_wrapper
from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase, WorkspaceError


class DockerWorkspace(WorkspaceBase):
    """A workspace whose commands run inside a Docker container.

    Args:
        mode: ``"exec"`` to reuse a long-lived container (pass ``container``),
            or ``"run"`` to start a fresh ``--rm`` container per command
            (pass ``image``).
        container: Container name/id, required when ``mode="exec"``.
        image: Image name, required when ``mode="run"``.
        workdir: Working directory inside the container (used as this
            workspace's ``root_path`` and as the ``cwd`` for commands).
        user: Optional ``-u`` user for ``exec`` mode.
        volumes: Optional bind mounts (``"src:dst"``) for ``run`` mode.
        env: Optional environment passthrough for ``run`` mode.
        network: Network mode for ``run`` mode (defaults to ``"none"``, same
            as :func:`~kinetic_sdk.terminal.docker.docker_run_wrapper`).
    """

    def __init__(
        self,
        *,
        mode: str = "exec",
        container: str | None = None,
        image: str | None = None,
        workdir: str = "/workspace",
        user: str | None = None,
        volumes: list[str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = "none",
    ) -> None:
        if mode not in ("exec", "run"):
            raise ValueError(f"mode must be 'exec' or 'run', got {mode!r}")
        self._mode = mode
        self._workdir = workdir
        if mode == "exec":
            if not container:
                raise ValueError("mode='exec' requires a container name")
            self._wrapper = docker_exec_wrapper(container, workdir=workdir, user=user)
        else:
            if not image:
                raise ValueError("mode='run' requires an image name")
            self._wrapper = docker_run_wrapper(
                image, workdir=workdir, volumes=volumes, env=env, network=network
            )

    @property
    def root_path(self) -> str:
        return self._workdir

    def run_command(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None
    ) -> CommandResult:
        """Build ``<wrapper> bash -c <command>`` and run it via subprocess.run,
        exactly the way ``TerminalTool.command_wrapper`` already does.
        """
        effective_command = command
        if cwd is not None:
            effective_command = f"cd {quote(cwd)} && {command}"
        argv = [*self._wrapper, "bash", "-c", effective_command]
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            return CommandResult(
                output="", exit_code=-1, duration_seconds=0.0, error=f"failed to start docker: {exc}"
            )
        timed_out = False
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            output, _ = proc.communicate()
        duration = time.monotonic() - started
        return CommandResult(
            output=output or "",
            exit_code=proc.returncode if proc.returncode is not None else -1,
            duration_seconds=round(duration, 3),
            timed_out=timed_out,
            error=(f"command timed out after {timeout}s" if timed_out else None),
        )

    def read_text(self, relative_path: str) -> str:
        if self._mode != "exec":
            raise WorkspaceError(
                "read_text is not supported without a persistent container "
                "(mode='run' starts a fresh container per command); use mode='exec'"
            )
        absolute = f"{self._workdir.rstrip('/')}/{relative_path.lstrip('/')}"
        result = self.run_command(f"cat {quote(absolute)}")
        if result.exit_code != 0:
            raise WorkspaceError(
                f"failed to read {relative_path!r} from container: {result.output}"
            )
        return result.output

    def write_text(self, relative_path: str, content: str) -> None:
        if self._mode != "exec":
            raise WorkspaceError(
                "write_text is not supported without a persistent container "
                "(mode='run' starts a fresh container per command); use mode='exec'"
            )
        absolute = f"{self._workdir.rstrip('/')}/{relative_path.lstrip('/')}"
        parent = absolute.rsplit("/", 1)[0] or "/"
        mkdir_result = self.run_command(f"mkdir -p {quote(parent)}")
        if mkdir_result.exit_code != 0:
            raise WorkspaceError(f"failed to create parent dir: {mkdir_result.output}")
        # run_command has no stdin wiring, so write via a single quoted
        # printf redirect rather than piping content through cat/heredoc.
        write_result = self.run_command(f"printf '%s' {quote(content)} > {quote(absolute)}")
        if write_result.exit_code != 0:
            raise WorkspaceError(f"failed to write {relative_path!r}: {write_result.output}")

    def delete_file(self, relative_path: str) -> None:
        if self._mode != "exec":
            raise WorkspaceError("delete_file is not supported without a persistent container")
        absolute = f"{self._workdir.rstrip('/')}/{relative_path.lstrip('/')}"
        result = self.run_command(f"rm -- {quote(absolute)}")
        if result.exit_code != 0:
            raise WorkspaceError(f"failed to delete {relative_path!r}: {result.output}")

    def list_files(self, pattern: str | None = None) -> list[str]:
        find_cmd = f"cd {quote(self._workdir)} && find . -type f"
        result = self.run_command(find_cmd)
        if result.exit_code != 0:
            raise WorkspaceError(f"failed to list files: {result.output}")
        files = [line[2:] for line in result.output.splitlines() if line.startswith("./")]
        if pattern is not None:
            import fnmatch

            files = [f for f in files if fnmatch.fnmatch(f, pattern)]
        return sorted(files)

    def list_directory(self, relative_path: str = ".") -> list[str]:
        directory = f"{self._workdir.rstrip('/')}/{relative_path.lstrip('./')}"
        result = self.run_command(f"find {quote(directory)} -mindepth 1 -maxdepth 1 -printf '%f%y\\n'")
        if result.exit_code != 0:
            raise WorkspaceError(f"failed to list directory: {result.output}")
        entries = []
        for line in result.output.splitlines():
            entries.append(line[:-1] + "/" if line.endswith("d") else line[:-1])
        return sorted(entries)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DockerWorkspace mode={self._mode!r} workdir={self._workdir!r}>"
