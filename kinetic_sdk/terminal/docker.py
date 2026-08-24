"""Docker sandbox adapters for ``TerminalTool``.

A policy gate stops the model from running ``rm -rf`` — but an APPROVED
command still runs with the full privileges of the agent host (same as
OpenHands/Claude sandboxing: the gate is policy, the sandbox is
isolation). ``docker_exec_wrapper`` / ``docker_run_wrapper`` build the argv
prefix that moves command execution into a container:

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
) -> list[str]:
    """Argv prefix running each command in a fresh ``--rm`` container.

    Defaults to ``network=none`` (no exfil via the sandbox), an arbitrary
    number of ``-v`` bind mounts and ``-e`` env passthrough. Empty image /
    malformed volume spec raises :class:`SandboxWrapperError`.
    """
    if not isinstance(image, str) or not image.strip():
        raise SandboxWrapperError("docker_run_wrapper needs a non-empty image")
    argv = ["docker", "run", "--rm", "-i"]
    if network is not None:
        argv += ["--network", network]
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
