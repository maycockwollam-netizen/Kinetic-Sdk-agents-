"""``RemoteAPIWorkspace``: forward run/read/write over HTTP to an agent-server.

Uses only :mod:`urllib.request` from the standard library — no new
dependency — matching the endpoint shapes
:class:`~kinetic_sdk.server.agent_server.AgentServer` already exposes
(``bearer`` auth header, JSON in/out). The remote side is expected to expose
three JSON endpoints under its base URL:

* ``POST {base_url}/workspace/run``   — ``{"command", "timeout", "cwd"}``
* ``POST {base_url}/workspace/read``  — ``{"path"}``
* ``POST {base_url}/workspace/write`` — ``{"path", "content"}``

An agent using this workspace cannot tell the difference from
:class:`~kinetic_sdk.workspace.manager.LocalWorkspace` — same
:class:`~kinetic_sdk.workspace.base.CommandResult` shape comes back either
way.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase, WorkspaceError


class RemoteAPIWorkspace(WorkspaceBase):
    """A workspace whose operations are proxied to a remote agent-server.

    Args:
        base_url: Server base URL, e.g. ``"http://localhost:8765"``. No
            trailing slash required.
        workspace_id: Identifier the remote server uses to route to the
            correct sandbox/session (sent as ``workspace_id`` in every call).
        token: Optional bearer token, sent as ``Authorization: Bearer <token>``.
        timeout: Default HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        workspace_id: str = "default",
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._workspace_id = workspace_id
        self._token = token
        self._http_timeout = timeout

    @property
    def root_path(self) -> str:
        return f"{self._base_url}#{self._workspace_id}"

    def _post(self, path: str, payload: dict, *, timeout: float | None = None) -> dict:
        body = json.dumps({"workspace_id": self._workspace_id, **payload}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(
                req, timeout=timeout or self._http_timeout
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise WorkspaceError(f"remote workspace error ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise WorkspaceError(f"could not reach remote workspace: {exc.reason}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkspaceError(f"remote workspace returned non-JSON response: {raw!r}") from exc

    def run_command(
        self, command: str, *, timeout: float | None = None, cwd: str | None = None
    ) -> CommandResult:
        data = self._post(
            "/workspace/run",
            {"command": command, "timeout": timeout, "cwd": cwd},
            timeout=(timeout + 5) if timeout else None,
        )
        return CommandResult(
            output=data.get("output", ""),
            exit_code=data.get("exit_code", -1),
            duration_seconds=data.get("duration_seconds", 0.0),
            timed_out=data.get("timed_out", False),
            truncated=data.get("truncated", False),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    def read_text(self, relative_path: str) -> str:
        data = self._post("/workspace/read", {"path": relative_path})
        if "content" not in data:
            raise WorkspaceError(data.get("error", f"failed to read {relative_path!r}"))
        return data["content"]

    def write_text(self, relative_path: str, content: str) -> None:
        data = self._post("/workspace/write", {"path": relative_path, "content": content})
        if not data.get("ok", False):
            raise WorkspaceError(data.get("error", f"failed to write {relative_path!r}"))

    def list_files(self, pattern: str | None = None) -> list[str]:
        data = self._post("/workspace/list", {"pattern": pattern})
        return list(data.get("files", []))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RemoteAPIWorkspace base_url={self._base_url!r} id={self._workspace_id!r}>"
