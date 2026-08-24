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
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        token: str | None = None,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self._agent_factory = agent_factory
        self._token = token
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
                if self.path != "/runs":
                    self._send_json(404, {"error": "not found"})
                    return
                body = self._read_body()
                if not body:
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
