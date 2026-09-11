"""Workspace package: scoped working-directory management for agents.

``Workspace`` (kept as a backward-compatible alias for
:class:`LocalWorkspace`) remains the default: path containment + subprocess
on the local filesystem. :class:`DockerWorkspace` and
:class:`RemoteAPIWorkspace` implement the same :class:`WorkspaceBase`
contract for containerised and remote execution — swap one for another
without touching agent code.
"""

from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase, WorkspaceError
from kinetic_sdk.workspace.docker_workspace import DockerWorkspace
from kinetic_sdk.workspace.manager import LocalWorkspace, PathTraversalError, Workspace
from kinetic_sdk.workspace.remote import RemoteAPIWorkspace

__all__ = [
    "CommandResult",
    "WorkspaceBase",
    "WorkspaceError",
    "PathTraversalError",
    "Workspace",
    "LocalWorkspace",
    "DockerWorkspace",
    "RemoteAPIWorkspace",
]
