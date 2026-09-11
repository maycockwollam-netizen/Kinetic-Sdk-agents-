"""Abstract workspace interface: run/read/write against *some* environment.

``WorkspaceBase`` is the contract every workspace implementation honours so
that :class:`~kinetic_sdk.terminal.tool.TerminalTool` and
:class:`~kinetic_sdk.files.tool.FileTool` never need to know whether they are
talking to the local filesystem, a Docker container, or a remote agent
server — they call ``run_command`` / ``read_text`` / ``write_text`` and get
back the same shapes regardless of where the work actually happens.

Concrete implementations live alongside this file:

* :class:`kinetic_sdk.workspace.manager.LocalWorkspace` (formerly the only
  ``Workspace`` — path containment + subprocess, unchanged behaviour).
* :class:`kinetic_sdk.workspace.docker_workspace.DockerWorkspace` — runs
  everything inside a container, reusing ``docker_exec_wrapper`` /
  ``docker_run_wrapper`` from ``terminal/docker.py``.
* :class:`kinetic_sdk.workspace.remote.RemoteAPIWorkspace` — forwards
  everything over HTTP to a remote ``AgentServer``.

Path safety (``resolve`` / traversal rejection) stays the job of
:class:`kinetic_sdk.workspace.manager.Workspace <kinetic_sdk.workspace.PathTraversalError>`-style
containment where a real local filesystem is involved; remote/docker
backends are responsible for enforcing their own boundaries server-side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResult:
    """Result of running a command through a workspace.

    Mirrors the fields :class:`~kinetic_sdk.terminal.tool.TerminalTool`
    already surfaces in its ``ToolResult.metadata``, so wiring a workspace's
    ``run_command`` into the tool is a straight field mapping.
    """

    output: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkspaceError(RuntimeError):
    """Raised when a workspace operation cannot be carried out.

    Used e.g. by a non-persistent :class:`DockerWorkspace` (``run`` mode,
    one throwaway container per command) for ``read_text``/``write_text``,
    which need a container that survives between calls.
    """


class WorkspaceBase(ABC):
    """Contract shared by every workspace backend.

    Subclasses decide *where* a command runs or a file lives; callers never
    branch on the concrete type — they call these methods and handle
    :class:`WorkspaceError` for operations a given backend cannot support.
    """

    @property
    @abstractmethod
    def root_path(self) -> str:
        """Identifier for the workspace root (local path, container path, or
        remote path) — used for display/logging, not necessarily a path on
        the machine Python is running on."""

    @abstractmethod
    def run_command(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None
    ) -> CommandResult:
        """Run *command* (shell semantics) and return its result.

        Args:
            command: Shell command, executed via ``bash -c``.
            timeout: Wall-clock seconds before the command is killed.
            cwd: Optional working directory *inside* the workspace, relative
                to the root. ``None`` uses the workspace root itself.
        """

    @abstractmethod
    def read_text(self, relative_path: str) -> str:
        """Read a text file inside the workspace.

        Raises:
            WorkspaceError: if this backend cannot service direct reads
                (e.g. a non-persistent Docker ``run`` workspace).
        """

    @abstractmethod
    def write_text(self, relative_path: str, content: str) -> None:
        """Write a text file inside the workspace, creating parents as needed.

        Raises:
            WorkspaceError: if this backend cannot service direct writes.
        """

    @abstractmethod
    def list_files(self, pattern: str | None = None) -> list[str]:
        """List files inside the workspace as sorted root-relative POSIX paths."""
