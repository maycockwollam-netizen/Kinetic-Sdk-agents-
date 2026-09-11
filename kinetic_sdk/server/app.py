"""``AgentServer``: run agents over a small REST API (stdlib only).

The SDK is a library; production usage needs a process that accepts "run
this message" over the network (OpenHands ships an Agent Server; ADK and
the OpenAI Agents runtime do the same). This module provides a minimal,
zero-dependency HTTP layer built on ``http.server`` so the core stays
stdlib-only — deploy behind your real gateway for TLS/load balancing.

API (JSON in/out):

* ``GET  /health`` -> ``{"status": "ok"}``
* ``POST /runs``   body ``{"message": str, "stream": bool?, "output_schema": dict?}``
    -> ``{"run_id": str, "status": "completed"|"error", "final": str,
    "structured_output": any|null, "usage": {...}, "error": str?}``
* ``GET  /runs/<run_id>`` -> the recorded run result (404 when unknown)
* ``POST /workspace/{run,read,write,delete,list,list-directory}`` -> managed
  workspace operations when ``workspace_factory`` is configured. Every body
  includes ``workspace_id``; the factory, not the HTTP client, selects the
  actual server-side workspace.

Every request spawns a fresh :class:`~kinetic_sdk.agent.agent.Agent` via
the injected ``agent_factory`` (conversations are per-run by design — the
caller that wants continuity hands the factory an agent built on a
``state_store``). Runs execute on the handler's thread;
``ThreadingHTTPServer`` keeps concurrent requests independent.

Optional bearer-token auth (``token=...``) checks ``Authorization: Bearer
<token>`` — a single shared secret for dev/staging; put a real auth layer
in front for anything exposed. Request bodies are capped (1 MB) so a
malicious client cannot memory-bomb the process.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

logger = logging.getLogger(__name__)

#: Hard cap on request bodies (bytes).
MAX_BODY_BYTES = 1024 * 1024


class RunRecord(dict):
    """One recorded run result (a plain dict; the alias documents intent)."""


class AgentServer:
    """HTTP front-end for one-shot agent runs.

    Args:
        agent_factory: Zero-arg callable returning a READY agent (tools,
            policy, memory... already configured). Called once per run.
        host/port: Bind address. ``port=0`` asks the OS for a free port
            (the effective port is readable on :attr:`port` after
            :meth:`start_in_thread`).
        token: Optional bearer token; when set, every non-``/health``
            request must present it.
        workspace_factory: Optional factory for the server-managed workspace
            identified by a client-supplied ``workspace_id``. When omitted,
            workspace endpoints are unavailable (404), rather than exposing
            the server host filesystem by accident.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        token: str | None = None,
        workspace_factory: Callable[[str], WorkspaceBase] | None = None,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self._agent_factory = agent_factory
        self._token = token
        self._workspace_factory = workspace_factory
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.host, self.port = self._httpd.server_address[:2]

    # --- lifecycle ------------------------------------------------------

    def serve_forever(self) -> None:  # pragma: no cover - blocking loop
        """Serve in the calling thread (use :meth:`start_in_thread` in apps)."""
        self._httpd.serve_forever()

    def start_in_thread(self, daemon: bool = True) -> threading.Thread:
        """Start the server on a background thread and return it."""
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=daemon)
        thread.start()
        return thread

    def shutdown(self) -> None:
        """Stop serving and close the socket."""
        self._httpd.shutdown()
        self._httpd.server_close()

    # --- run execution --------------------------------------------------

    def _run_agent(self, message: str, stream: bool, output_schema: dict | None) -> RunRecord:
        run_id = uuid.uuid4().hex
        record = RunRecord(run_id=run_id, status="running", final="")
        with self._lock:
            self._runs[run_id] = record
        try:
            agent = self._agent_factory()
            final = agent.run(message, stream=stream, output_schema=output_schema)
            record.update(
                status="completed",
                final=final,
                structured_output=agent.structured_output,
                usage=agent.usage.snapshot().to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - never leak a stack over HTTP
            logger.exception("Run %s failed", run_id)
            record.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        with self._lock:
            self._runs[run_id] = record
        return record

    # --- HTTP plumbing --------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                logger.debug("%s - %s", self.address_string(), format % args)

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                if server._token is None:
                    return True
                expected = f"Bearer {server._token}"
                return self.headers.get("Authorization") == expected

            def _read_body(self) -> dict[str, Any] | None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return None
                if length > MAX_BODY_BYTES:
                    self._send_json(413, {"error": "request body too large"})
                    return {}
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": "body must be JSON"})
                    return {}
                if not isinstance(data, dict):
                    self._send_json(400, {"error": "body must be a JSON object"})
                    return {}
                return data

            def _workspace(self, body: dict[str, Any]) -> WorkspaceBase | None:
                if server._workspace_factory is None:
                    self._send_json(404, {"error": "workspace API is not configured"})
                    return None
                workspace_id = body.get("workspace_id", "default")
                if not isinstance(workspace_id, str) or not workspace_id:
                    self._send_json(400, {"error": "'workspace_id' must be a non-empty string"})
                    return None
                try:
                    return server._workspace_factory(workspace_id)
                except (ValueError, WorkspaceError) as exc:
                    self._send_json(404, {"error": f"workspace not found: {exc}"})
                    return None

            def _workspace_post(self, path: str, body: dict[str, Any]) -> bool:
                workspace = self._workspace(body)
                if workspace is None:
                    return False
                try:
                    if path == "/workspace/run":
                        command = body.get("command")
                        timeout = body.get("timeout")
                        cwd = body.get("cwd")
                        if not isinstance(command, str) or not command.strip():
                            self._send_json(400, {"error": "'command' must be a non-empty string"})
                            return True
                        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
                            self._send_json(400, {"error": "'timeout' must be a positive number"})
                            return True
                        if cwd is not None and not isinstance(cwd, str):
                            self._send_json(400, {"error": "'cwd' must be a string or null"})
                            return True
                        result = workspace.run_command(command, timeout=timeout, cwd=cwd)
                        self._send_json(200, {
                            "output": result.output, "exit_code": result.exit_code,
                            "duration_seconds": result.duration_seconds,
                            "timed_out": result.timed_out, "truncated": result.truncated,
                            "error": result.error, "metadata": result.metadata,
                        })
                    elif path == "/workspace/read":
                        file_path = body.get("path")
                        if not isinstance(file_path, str) or not file_path:
                            self._send_json(400, {"error": "'path' must be a non-empty string"})
                            return True
                        self._send_json(200, {"content": workspace.read_text(file_path)})
                    elif path == "/workspace/write":
                        file_path, content = body.get("path"), body.get("content")
                        if not isinstance(file_path, str) or not file_path or not isinstance(content, str):
                            self._send_json(400, {"error": "'path' and 'content' must be strings"})
                            return True
                        workspace.write_text(file_path, content)
                        self._send_json(200, {"ok": True})
                    elif path == "/workspace/delete":
                        file_path = body.get("path")
                        if not isinstance(file_path, str) or not file_path:
                            self._send_json(400, {"error": "'path' must be a non-empty string"})
                            return True
                        workspace.delete_file(file_path)
                        self._send_json(200, {"ok": True})
                    elif path == "/workspace/list":
                        pattern = body.get("pattern")
                        if pattern is not None and not isinstance(pattern, str):
                            self._send_json(400, {"error": "'pattern' must be a string or null"})
                            return True
                        self._send_json(200, {"files": workspace.list_files(pattern)})
                    else:  # /workspace/list-directory
                        file_path = body.get("path", ".")
                        if not isinstance(file_path, str):
                            self._send_json(400, {"error": "'path' must be a string"})
                            return True
                        self._send_json(200, {"entries": workspace.list_directory(file_path)})
                except FileNotFoundError as exc:
                    self._send_json(404, {"error": str(exc)})
                except (OSError, ValueError, WorkspaceError) as exc:
                    self._send_json(400, {"error": str(exc)})
                return True

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                    return
                if not self._authed():
                    self._send_json(401, {"error": "unauthorized"})
                    return
                if self.path.startswith("/runs/"):
                    run_id = self.path.rsplit("/", 1)[-1]
                    with server._lock:
                        record = server._runs.get(run_id)
                    if record is None:
                        self._send_json(404, {"error": "run not found"})
                        return
                    self._send_json(200, dict(record))
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authed():
                    self._send_json(401, {"error": "unauthorized"})
                    return
                workspace_paths = {
                    "/workspace/run", "/workspace/read", "/workspace/write",
                    "/workspace/delete", "/workspace/list", "/workspace/list-directory",
                }
                if self.path != "/runs" and self.path not in workspace_paths:
                    self._send_json(404, {"error": "not found"})
                    return
                body = self._read_body()
                if not body:
                    return
                if self.path in workspace_paths:
                    self._workspace_post(self.path, body)
                    return
                message = body.get("message")
                if not isinstance(message, str) or not message.strip():
                    self._send_json(400, {"error": "'message' must be a non-empty string"})
                    return
                schema = body.get("output_schema")
                if schema is not None and not isinstance(schema, dict):
                    self._send_json(400, {"error": "'output_schema' must be an object"})
                    return
                record = server._run_agent(
                    message,
                    stream=bool(body.get("stream", False)),
                    output_schema=schema,
                )
                self._send_json(200 if record["status"] == "completed" else 500, dict(record))

        return Handler
