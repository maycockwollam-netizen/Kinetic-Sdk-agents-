"""Docker sandbox adapters for ``TerminalTool``.

A policy gate stops the model from running ``rm -rf`` — but an APPROVED
command still needs isolation from the agent host. ``docker_run_wrapper``
therefore creates fresh containers with networking disabled plus a read-only
root filesystem, no Linux capabilities, and no privilege escalation by
default. These controls reduce container privileges but do not make unsafe
mounts, a privileged Docker daemon, or an untrusted image safe on their own.
``docker_exec_wrapper`` / ``docker_run_wrapper`` build the argv prefix that
moves command execution into a container:

* ``exec`` — reuse a long-lived container you started yourself
  (matches dev-container workflows: the repo is mounted, services up).
* ``run`` — one fresh container per command ("--rm"), maximal isolation;
  each command starts from a clean filesystem.

Both are plain argv builders — ``TerminalTool`` still handles
timeout/process-group kill, so a timed-out ``docker exec`` command dies
alongside its in-container bash (kill the group on the host, docker relays
SIGKILL into the container for exec'd processes).
"""

from __future__ import annotations

import re


class SandboxWrapperError(ValueError):
    """Raised when a container/image/workdir value is malformed."""


_SAFE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")


def docker_exec_wrapper(
    container: str, *, workdir: str | None = None, user: str | None = None
) -> list[str]:
    """Argv prefix executing commands inside a running container.

    Prepends ``docker exec [-w workdir] [-u user] -i <container>`` so
    :class:`~kinetic_sdk.terminal.tool.TerminalTool` executes
    ``<prefix> bash -c <command>`` inside the container.

    Container hardening (capability dropping, read-only filesystems, and
    resource limits) must be configured when the externally managed
    container is created. ``docker_exec_wrapper`` cannot and should not
    attempt to impose those creation-time limits again.
    """
    if not _SAFE.match(container):
        raise SandboxWrapperError(f"invalid container name: {container!r}")
    argv = ["docker", "exec"]
    if workdir is not None:
        argv += ["-w", workdir]
    if user is not None:
        argv += ["-u", user]
    argv += ["-i", container]
    return argv


def docker_run_wrapper(
    image: str,
    *,
    workdir: str | None = None,
    volumes: list[str] | None = None,
    env: dict[str, str] | None = None,
    network: str | None = "none",
    read_only: bool = True,
    drop_all_capabilities: bool = True,
    no_new_privileges: bool = True,
    memory_limit: str | None = None,
    cpu_limit: float | None = None,
    pids_limit: int | None = None,
    user: str | None = None,
) -> list[str]:
    """Argv prefix running each command in a fresh ``--rm`` container.

    Defaults to no network, a read-only root filesystem, no Linux
    capabilities, and no privilege escalation. ``memory_limit``,
    ``cpu_limit``, ``pids_limit``, and ``user`` are opt-in because their
    appropriate values depend on the caller. Writable work should be placed
    in explicitly supplied writable volumes. Empty image / malformed volume
    spec raises :class:`SandboxWrapperError`.
    """
    if not isinstance(image, str) or not image.strip():
        raise SandboxWrapperError("docker_run_wrapper needs a non-empty image")
    argv = ["docker", "run", "--rm", "-i"]
    if network is not None:
        argv += ["--network", network]
    if read_only:
        argv.append("--read-only")
    if drop_all_capabilities:
        argv.append("--cap-drop=ALL")
    if no_new_privileges:
        argv.append("--security-opt=no-new-privileges:true")
    if memory_limit is not None:
        argv += ["--memory", memory_limit]
    if cpu_limit is not None:
        argv += ["--cpus", str(cpu_limit)]
    if pids_limit is not None:
        argv += ["--pids-limit", str(pids_limit)]
    if user is not None:
        argv += ["--user", user]
    for vol in volumes or []:
        if ":" not in vol:
            raise SandboxWrapperError(f"volume spec needs 'src:dst', got {vol!r}")
        argv += ["-v", vol]
    for key in env or {}:
        argv += ["-e", key]
    if workdir is not None:
        argv += ["-w", workdir]
    argv.append(image)
    return argv
