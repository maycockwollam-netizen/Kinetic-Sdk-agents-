"""Workspace management: one scoped working directory per agent (Stage 4, part B).

A :class:`Workspace` represents the single directory an agent is allowed to
touch. Its job is to make *where* file operations happen explicit and safe,
instead of letting every tool manage its own paths ad hoc:

* :meth:`Workspace.resolve` turns a relative path into an absolute one and
  **refuses anything that escapes the root** — ``../`` segments, absolute
  paths outside the root, and symlinks whose targets live outside the root
  all raise :class:`PathTraversalError`. This is the SDK's defence against
  path-traversal payloads in LLM-generated tool input (e.g. a model asked to
  read ``../../../etc/passwd``).
* :meth:`Workspace.list_files` enumerates the files inside the root
  (optionally filtered by a simple glob-style pattern) so an agent can orient
  itself without shelling out to ``ls``/``find``.

This version deliberately keeps the model simple: one agent, one workspace.
Multi-workspace support is a later-version concern.

Adoption is incremental: other tools (e.g. :class:`kinetic_sdk.git.GitTool`
via ``GitTool(workdir=workspace.root_path)``, or a future file-system tool)
*may* route their paths through a workspace, but existing tools are not
required to change.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase, WorkspaceError


class PathTraversalError(ValueError):
    """Raised when a path would resolve outside the workspace root.

    The message names the offending input so the agent (and the audit trail)
    can see exactly which path was rejected.
    """


class LocalWorkspace(WorkspaceBase):
    """A single working directory an agent operates inside.

    Args:
        root_path: The workspace root. Must be an existing directory; it is
            canonicalised (symlinks resolved) once at construction so every
            later containment check compares real paths.

    Attributes:
        root_path: Canonical absolute path of the workspace root.
    """

    def __init__(self, root_path: str | os.PathLike[str]) -> None:
        root = os.path.realpath(os.fspath(root_path))
        if not os.path.isdir(root):
            raise ValueError(f"workspace root is not an existing directory: {root_path!r}")
        self._root = root

    @property
    def root_path(self) -> str:
        """Canonical absolute path of the workspace root."""
        return self._root

    def resolve(self, relative_path: str) -> str:
        """Resolve *relative_path* against the root, rejecting escapes.

        The candidate is canonicalised with :func:`os.path.realpath` (so
        ``a/../../b`` collapses and symlinks are followed) and must end up
        inside the root. Anything else — ``../`` traversal, absolute paths
        outside the root, symlinks pointing outside — raises
        :class:`PathTraversalError`.

        Args:
            relative_path: Path relative to the root. An absolute path is
                accepted only when it already points inside the root.

        Returns:
            The canonical absolute path inside the workspace.
        """
        candidate = os.path.realpath(os.path.join(self._root, os.fspath(relative_path)))
        try:
            inside = os.path.commonpath([self._root, candidate]) == self._root
        except ValueError:
            # Mixed drives (Windows) or otherwise incomparable paths.
            inside = False
        if not inside:
            raise PathTraversalError(
                f"path {relative_path!r} resolves outside the workspace root {self._root!r}"
            )
        return candidate

    def list_files(self, pattern: str | None = None) -> list[str]:
        """List files inside the workspace, as sorted root-relative paths.

        Args:
            pattern: Optional filter matched with :func:`fnmatch.fnmatch`
                against each root-relative POSIX path. ``*`` crosses directory
                separators here, so ``"*.py"`` also matches ``src/a.py``.
                ``None`` lists every file.

        Returns:
            Sorted list of relative POSIX paths (files only, no directories).
        """
        matches: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(self._root):
            for filename in filenames:
                absolute = os.path.join(dirpath, filename)
                relative = os.path.relpath(absolute, self._root)
                posix = Path(relative).as_posix()
                if pattern is None or fnmatch.fnmatch(posix, pattern):
                    matches.append(posix)
        return sorted(matches)

    def list_directory(self, relative_path: str = ".") -> list[str]:
        directory = Path(self.resolve(relative_path))
        if not directory.is_dir():
            raise NotADirectoryError(relative_path)
        return sorted(entry.name + ("/" if entry.is_dir() else "") for entry in directory.iterdir())

    def run_command(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None
    ) -> CommandResult:
        """Refuse shell execution on the SDK host.

        ``cwd`` confines only a process's initial directory; it cannot stop
        shell text from using absolute paths, ``cd ..``, redirects, or child
        processes to access the host.  Therefore a local workspace is a
        path-safe *file* abstraction, not a command sandbox.  Use
        :class:`DockerWorkspace` (or a Kubernetes/remote workspace supplied
        by the platform) when untrusted command text must be executed.
        """
        del command, timeout, cwd
        raise WorkspaceError(
            "LocalWorkspace does not execute shell commands because it is not a sandbox; "
            "use DockerWorkspace, KubernetesWorkspace, or RemoteAPIWorkspace"
        )

    def read_text(self, relative_path: str) -> str:
        """Read a text file inside the workspace root."""
        return Path(self.resolve(relative_path)).read_text()

    def write_text(self, relative_path: str, content: str) -> None:
        """Write a text file inside the workspace root, creating parents as needed."""
        target = Path(self.resolve(relative_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def delete_file(self, relative_path: str) -> None:
        """Delete a file in the workspace, rejecting directories."""
        target = Path(self.resolve(relative_path))
        if target.is_dir():
            raise IsADirectoryError(relative_path)
        target.unlink()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LocalWorkspace root={self._root!r}>"


#: Backward-compatible alias — existing imports of ``Workspace`` keep working
#: unchanged; it is exactly ``LocalWorkspace`` under a familiar name.
Workspace = LocalWorkspace
