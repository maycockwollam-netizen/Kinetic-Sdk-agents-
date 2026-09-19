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

import os
import re


class SandboxWrapperError(ValueError):
    """Raised when a container/image/workdir value is malformed."""


_SAFE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")
_SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)?$")
_MEMORY_LIMIT = re.compile(r"^\d+[bkmg]?$", re.IGNORECASE)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_MOUNT_SOURCES = {
    "/",
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/var/run/docker.sock",
    "/run/docker.sock",
}


def _validate_volume(volume: str, *, allow_unsafe_mounts: bool) -> None:
    """Reject malformed and sensitive host bind mounts by default."""
    if not isinstance(volume, str) or ":" not in volume:
        raise SandboxWrapperError(f"volume spec needs 'src:dst', got {volume!r}")
    source = volume.split(":", 1)[0]
    if not source:
        raise SandboxWrapperError(f"volume spec needs a source path, got {volume!r}")
    # Resolve dot segments and symlinks before comparing, so aliases such as
    # ``/workspace/../etc`` cannot bypass the sensitive-path guard.
    normalised_source = os.path.realpath(os.path.abspath(source))
    is_unsafe = normalised_source in _UNSAFE_MOUNT_SOURCES or any(
        normalised_source.startswith(f"{directory}/")
        for directory in ("/proc", "/sys", "/dev", "/etc")
    )
    if not allow_unsafe_mounts and is_unsafe:
        raise SandboxWrapperError(f"unsafe mount source: {source!r}")


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
    tmpfs: list[str] | None = None,
    drop_all_capabilities: bool = True,
    no_new_privileges: bool = True,
    memory_limit: str | None = None,
    cpu_limit: float | None = None,
    pids_limit: int | None = None,
    user: str | None = None,
    allow_unsafe_mounts: bool = False,
) -> list[str]:
    """Argv prefix running each command in a fresh ``--rm`` container.

    Defaults to no network, a read-only root filesystem, no Linux
    capabilities, and no privilege escalation. ``memory_limit``,
    ``cpu_limit``, ``pids_limit``, and ``user`` are opt-in because their
    appropriate values depend on the caller. A read-only root gets a writable
    ``/tmp`` tmpfs by default; pass ``tmpfs=[]`` to disable it or supply a
    list of tmpfs mount specifications. Writable work should otherwise be
    placed in explicitly supplied writable volumes. Sensitive host mount
    sources are rejected unless ``allow_unsafe_mounts=True``. Malformed
    inputs raise :class:`SandboxWrapperError`.
    """
    if (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or any(char.isspace() for char in image)
    ):
        raise SandboxWrapperError("docker_run_wrapper needs a non-empty image")
    if workdir is not None and (not isinstance(workdir, str) or not os.path.isabs(workdir)):
        raise SandboxWrapperError(f"workdir must be an absolute path, got {workdir!r}")
    if user is not None and (not isinstance(user, str) or not _SAFE_USER.fullmatch(user)):
        raise SandboxWrapperError(f"invalid user: {user!r}")
    if memory_limit is not None and (
        not isinstance(memory_limit, str) or not _MEMORY_LIMIT.fullmatch(memory_limit)
    ):
        raise SandboxWrapperError(f"invalid memory limit: {memory_limit!r}")
    if cpu_limit is not None and (
        isinstance(cpu_limit, bool) or not isinstance(cpu_limit, (int, float)) or cpu_limit <= 0
    ):
        raise SandboxWrapperError(f"cpu_limit must be greater than zero, got {cpu_limit!r}")
    if pids_limit is not None and (
        isinstance(pids_limit, bool) or not isinstance(pids_limit, int) or pids_limit < 1
    ):
        raise SandboxWrapperError(f"pids_limit must be at least one, got {pids_limit!r}")
    for key in (env or {}):
        if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
            raise SandboxWrapperError(f"invalid environment variable name: {key!r}")
    argv = ["docker", "run", "--rm", "-i"]
    if network is not None:
        argv += ["--network", network]
    if read_only:
        argv.append("--read-only")
    effective_tmpfs = ["/tmp"] if read_only and tmpfs is None else (tmpfs or [])
    for mount in effective_tmpfs:
        if not isinstance(mount, str) or not mount:
            raise SandboxWrapperError(f"invalid tmpfs mount: {mount!r}")
        argv += ["--tmpfs", mount]
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
        _validate_volume(vol, allow_unsafe_mounts=allow_unsafe_mounts)
        argv += ["-v", vol]
    for key in env or {}:
        argv += ["-e", key]
    if workdir is not None:
        argv += ["-w", workdir]
    argv.append(image)
    return argv
