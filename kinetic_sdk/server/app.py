"""``AgentServer``: run agents over a small REST API (stdlib only).

The SDK is a library; production usage needs a process that accepts "run
this message" over the network (OpenHands ships an Agent Server; ADK and
the OpenAI Agents runtime do the same). This module provides a minimal,
zero-dependency HTTP layer built on ``http.server`` so the core stays
stdlib-only — deploy behind your real gateway for TLS/load balancing.

API (JSON in/out):

* ``GET  /health`` -> ``{"status": "ok"}``
* ``POST /runs``   body ``{"message": str, "session_id": str?, "stream": bool?, "output_schema": dict?}``
    -> ``{"run_id": str, "status": "completed"|"error", "final": str,
    "structured_output": any|null, "usage": {...}, "error": str?}``
* ``GET  /runs/<run_id>`` -> the recorded run result (404 when unknown)
* ``POST /v1/chat/completions`` -> small OpenAI-compatible gateway; its
  ``model`` field is used as the Kinetic session id, not as LLM model
  selection. ``stream: true`` returns an SSE ``text/event-stream`` of
  ``chat.completion.chunk`` objects ending in ``[DONE]``; otherwise a single
  ``chat.completion`` JSON object.
* ``GET /runs/stream`` -> WebSocket endpoint. The connection stays open and
  accepts any number of JSON text frames, each with ``message`` (and optional
  ``session_id``); every request streams its run events as text frames
  followed by a ``run.completed`` frame. Ping/pong and close frames are
  handled transparently, fragmented text messages are reassembled, and data
  payloads are capped at ``MAX_WS_PAYLOAD_BYTES``.
* ``POST /workspace/{run,read,write,delete,list,list-directory}`` -> managed
  workspace operations when ``workspace_factory`` is configured. Every body
  includes ``workspace_id``; the factory, not the HTTP client, selects the
  actual server-side workspace.

Requests without ``session_id`` spawn a fresh
:class:`~kinetic_sdk.agent.agent.Agent`. Session requests reuse an agent from
an in-memory cache, preserving its conversation until its configured TTL
expires. The factory may accept ``session_id`` but legacy zero-argument
factories remain supported. Runs execute on the handler's thread;
``ThreadingHTTPServer`` keeps concurrent requests independent.

Optional bearer-token auth (``token=...``) checks ``Authorization: Bearer
<token>`` — a single shared secret for dev/staging; put a real auth layer
in front for anything exposed. Request bodies are capped (1 MB) so a
malicious client cannot memory-bomb the process.

This is intentionally a small embedded HTTP server, not an Internet-facing
gateway: TLS and identity verification belong at a reverse proxy.  Its
in-memory run history and concurrent handlers are bounded to avoid turning
normal production traffic into unbounded process resource use.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import os
import socket
import struct
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.workspace.base import WorkspaceBase, WorkspaceError

logger = logging.getLogger(__name__)

#: Hard cap on request bodies (bytes).
MAX_BODY_BYTES = 1024 * 1024

#: Hard cap on a single WebSocket message payload (bytes). Control frames
#: (ping/pong/close) are exempt; oversized data frames close the connection.
MAX_WS_PAYLOAD_BYTES = 8 * 1024 * 1024


class RunRecord(dict):
    """One recorded run result (a plain dict; the alias documents intent)."""


class AgentServer:
    """HTTP front-end for one-shot agent runs.

    Args:
        agent_factory: Callable returning a READY agent (tools, policy,
            memory... already configured). It may accept ``session_id``;
            legacy zero-argument callables remain supported.
        host/port: Bind address. ``port=0`` asks the OS for a free port
            (the effective port is readable on :attr:`port` after
            :meth:`start_in_thread`).
        token: Optional bearer token; when set, every non-``/health``
            request must present it.
        workspace_factory: Optional factory for the server-managed workspace
            identified by a client-supplied ``workspace_id``. When omitted,
            workspace endpoints are unavailable (404), rather than exposing
            the server host filesystem by accident.
        session_ttl: Idle seconds after which cached session agents are
            discarded. ``None`` keeps sessions until server shutdown.
        max_requests_per_minute: Optional per-client-IP sliding-window limit.
            ``None`` keeps the existing unlimited behaviour.
        max_runs: Maximum completed/running run records retained in memory.
            The oldest records are evicted when the bound is reached.
        max_concurrent_requests: Maximum active HTTP handlers. Excess
            requests receive ``503`` rather than creating unbounded threads.
    """

    def __init__(
        self,
        agent_factory: Callable[..., Agent],
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        token: str | None = None,
        workspace_factory: Callable[[str], WorkspaceBase] | None = None,
        session_ttl: float | None = None,
        max_requests_per_minute: int | None = None,
        max_runs: int = 1_000,
        max_concurrent_requests: int = 64,
        session_store_dir: str | os.PathLike | None = None,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self._agent_factory = agent_factory
        self._token = token
        self._workspace_factory = workspace_factory
        if session_ttl is not None and session_ttl <= 0:
            raise ValueError("session_ttl must be positive or None")
        if max_requests_per_minute is not None and max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive or None")
        if max_runs <= 0:
            raise ValueError("max_runs must be positive")
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        self._session_ttl = session_ttl
        self._runs: dict[str, RunRecord] = {}
        self._run_order: deque[str] = deque()
        self._max_runs = max_runs
        self._sessions: dict[str, Agent] = {}
        self._session_activity: dict[str, float] = {}
        #: Per-session run locks. ``_lock`` still guards the session maps
        #: themselves; only the cached Agent's run() is serialised per session
        #: so a long run in one session never blocks another session's run.
        self._session_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        #: Optional on-disk store for run history. When set, run records are
        #: appended to ``<dir>/runs.jsonl`` and reloaded on startup, so a
        #: restart keeps the run index durable. Live Agent sessions are not
        #: serialized — wire the agent factory's ``state_store`` for full
        #: conversation persistence.
        self._store_dir: Path | None = None
        if session_store_dir is not None:
            self._store_dir = Path(session_store_dir)
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._load_stored_runs()
        self._drain_condition = threading.Condition(self._lock)
        self._in_flight_requests = 0
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._stopping = False
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times: dict[str, deque[float]] = {}
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

    def stop(self, timeout: float = 10.0) -> None:
        """Drain active handlers for up to *timeout* seconds before stopping.

        Handlers accepted before the drain marker finish normally; later
        requests receive ``503`` instead of extending shutdown indefinitely.
        Python cannot safely preempt an agent, so timeout is best-effort.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._drain_condition:
            self._stopping = True
            while self._in_flight_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("AgentServer shutdown drain timed out with %d in-flight request(s)", self._in_flight_requests)
                    break
                self._drain_condition.wait(remaining)
        self._httpd.shutdown()
        self._httpd.server_close()

    def shutdown(self) -> None:
        """Backward-compatible alias for :meth:`stop`."""
        self.stop()

    def _begin_request(self) -> tuple[bool, str | None]:
        with self._drain_condition:
            if self._stopping:
                return False, "server is shutting down"
            if not self._request_slots.acquire(blocking=False):
                return False, "server is at concurrent request capacity"
            self._in_flight_requests += 1
            return True, None

    def _finish_request(self) -> None:
        with self._drain_condition:
            self._in_flight_requests -= 1
            self._request_slots.release()
            self._drain_condition.notify_all()

    def _health_snapshot(self) -> tuple[int, int]:
        with self._lock:
            self._purge_expired_sessions()
            return len(self._sessions), self._in_flight_requests

    def _allow_request(self, client_ip: str) -> tuple[bool, int]:
        """Apply the optional in-memory per-IP sliding-window limiter."""
        if self._max_requests_per_minute is None:
            return True, 0
        now = time.monotonic()
        with self._lock:
            times = self._request_times.setdefault(client_ip, deque())
            while times and times[0] <= now - 60.0:
                times.popleft()
            if len(times) >= self._max_requests_per_minute:
                return False, max(1, int(60.0 - (now - times[0])) + 1)
            times.append(now)
            return True, 0

    # --- run execution --------------------------------------------------

    def _call_factory(self, session_id: str | None) -> Agent:
        """Call a session-aware factory, retaining legacy zero-arg support."""
        try:
            inspect.signature(self._agent_factory).bind(session_id)
        except (TypeError, ValueError):
            return self._agent_factory()
        return self._agent_factory(session_id)

    def _purge_expired_sessions(self) -> None:
        """Delete idle cached agents; caller must hold ``_lock``."""
        if self._session_ttl is None:
            return
        cutoff = time.monotonic() - self._session_ttl
        for session_id, active_at in list(self._session_activity.items()):
            if active_at < cutoff:
                self._sessions.pop(session_id, None)
                self._session_activity.pop(session_id, None)
                self._session_locks.pop(session_id, None)

    def _session_lock(self, session_id: str) -> threading.Lock:
        """Return the run lock for *session_id*, creating it if needed."""
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _make_agent(self, session_id: str | None) -> Agent:
        if session_id is None:
            return self._call_factory(None)
        with self._lock:
            self._purge_expired_sessions()
            agent = self._sessions.get(session_id)
            if agent is None:
                agent = self._call_factory(session_id)
                self._sessions[session_id] = agent
            self._session_activity[session_id] = time.monotonic()
            return agent

    def _run_agent(
        self,
        message: str,
        stream: bool,
        output_schema: dict | None,
        session_id: str | None = None,
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        record = RunRecord(run_id=run_id, status="running", final="")
        with self._lock:
            while len(self._run_order) >= self._max_runs:
                self._runs.pop(self._run_order.popleft(), None)
            self._runs[run_id] = record
            self._run_order.append(run_id)
        try:
            agent = self._make_agent(session_id)
            # ConversationState is mutable; serialise session runs PER SESSION
            # (never globally) so a long run in one session cannot block a
            # different session's run. Two runs on the SAME cached agent are
            # still serialised by that session's lock.
            if session_id is None:
                final = agent.run(message, stream=stream, output_schema=output_schema)
            else:
                with self._session_lock(session_id):
                    final = agent.run(message, stream=stream, output_schema=output_schema)
                with self._lock:
                    self._session_activity[session_id] = time.monotonic()
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
        self._persist_run(record)
        return record

    def _run_agent_streaming(
        self,
        message: str,
        output_schema: dict | None,
        session_id: str | None,
        writer: Callable[[str, dict[str, Any]], None],
    ) -> RunRecord:
        """Run an agent while streaming its events to *writer*.

        *writer* is invoked synchronously for every agent bus event (the same
        shape the event bus delivers, so the caller decides what is SSE data
        vs. an event boundary). Returns the same :class:`RunRecord` as
        :meth:`_run_agent`.
        """
        from kinetic_sdk.event.bus import Event

        run_id = uuid.uuid4().hex
        record = RunRecord(run_id=run_id, status="running", final="")
        with self._lock:
            while len(self._run_order) >= self._max_runs:
                self._runs.pop(self._run_order.popleft(), None)
            self._runs[run_id] = record
            self._run_order.append(run_id)

        agent = self._make_agent(session_id)

        def _stream(event: Event) -> None:
            writer(event.type, event.payload or {})

        agent.event_bus.subscribe("*", _stream)
        try:
            if session_id is None:
                final = agent.run(message, stream=True, output_schema=output_schema)
            else:
                with self._session_lock(session_id):
                    final = agent.run(message, stream=True, output_schema=output_schema)
                with self._lock:
                    self._session_activity[session_id] = time.monotonic()
            record.update(
                status="completed",
                final=final,
                structured_output=agent.structured_output,
                usage=agent.usage.snapshot().to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 - never leak a stack over HTTP
            logger.exception("Run %s failed", run_id)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            agent.event_bus.unsubscribe("*", _stream)
        with self._lock:
            self._runs[run_id] = record
        self._persist_run(record)
        return record

    # -- persistence ------------------------------------------------------

    def _runs_path(self) -> Path:
        """Path to the append-only run log when a store directory is set."""
        assert self._store_dir is not None
        return self._store_dir / "runs.jsonl"

    def _load_stored_runs(self) -> None:
        """Rebuild the run index from the on-disk append-only log."""
        if self._store_dir is None:
            return
        path = self._runs_path()
        if not path.exists():
            return
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        run_id = rec.get("run_id")
                        if not isinstance(run_id, str):
                            continue
                        self._runs[run_id] = RunRecord(rec)
                        self._run_order.append(run_id)
            except OSError:
                logger.exception("Failed to load stored runs from %s", path)

    def _persist_run(self, record: RunRecord) -> None:
        """Append one run record to the on-disk log (best-effort)."""
        if self._store_dir is None:
            return
        try:
            with open(self._runs_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.exception("Failed to persist run %s", record.get("run_id"))

    # --- HTTP plumbing --------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            _response_status: int | None = None

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                logger.debug("%s - %s", self.address_string(), format % args)

            def _send_json(
                self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None
            ) -> None:
                self._response_status = status
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _handle_request(self, operation: Callable[[], None]) -> None:
                """Count, limit and log one parsed HTTP request."""
                started = time.monotonic()
                self._response_status = None
                accepted, reason = server._begin_request()
                if not accepted:
                    self._send_json(503, {"error": reason})
                    self._log_request(started)
                    return
                try:
                    allowed, retry_after = server._allow_request(self.client_address[0])
                    if not allowed:
                        logger.warning("AgentServer rate limited request from %s to %s", self.client_address[0], self.path)
                        self._send_json(429, {"error": "rate_limited"}, {"Retry-After": str(retry_after)})
                        return
                    operation()
                finally:
                    server._finish_request()
                    self._log_request(started)

            def _log_request(self, started: float) -> None:
                logger.info(
                    "agent_server_request method=%s path=%s status=%s duration_seconds=%.3f",
                    self.command,
                    self.path,
                    self._response_status if self._response_status is not None else 500,
                    time.monotonic() - started,
                )

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
                self._handle_request(self._do_get)

            def _do_get(self) -> None:
                if self.path == "/health":
                    sessions, in_flight = server._health_snapshot()
                    self._send_json(200, {"status": "ok", "active_sessions": sessions, "in_flight_requests": in_flight})
                    return
                if not self._authed():
                    self._send_json(401, {"error": "unauthorized"})
                    return
                if self.path == "/runs/stream":
                    self._websocket_run()
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
                self._handle_request(self._do_post)

            def _sse_run(
                self, message: str, schema: dict[str, Any] | None, session_id: str | None
            ) -> None:
                """Stream one run's agent events as a Server-Sent Events stream.

                Each agent bus event becomes an ``event: <type>`` / ``data:
                <json payload>`` pair; the stream ends with an ``event: done``
                carrying the final :class:`RunRecord`.
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self._response_status = 200
                self.close_connection = True

                def _write(event_type: str, payload: dict[str, Any]) -> None:
                    data = json.dumps(payload, ensure_ascii=False, default=str)
                    self.wfile.write(
                        f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()

                try:
                    record = server._run_agent_streaming(
                        message, schema, session_id, _write
                    )
                    _write("done", dict(record))
                except Exception as exc:  # noqa: BLE001 - never leak over HTTP
                    logger.exception("SSE run failed")
                    _write("error", {"error": f"{type(exc).__name__}: {exc}"})
                finally:
                    try:
                        self.wfile.flush()
                    except Exception:  # noqa: BLE001 - best effort
                        pass

            def _do_post(self) -> None:
                if not self._authed():
                    self._send_json(401, {"error": "unauthorized"})
                    return
                workspace_paths = {
                    "/workspace/run", "/workspace/read", "/workspace/write",
                    "/workspace/delete", "/workspace/list", "/workspace/list-directory",
                }
                if self.path not in {"/runs", "/v1/chat/completions"} and self.path not in workspace_paths:
                    self._send_json(404, {"error": "not found"})
                    return
                body = self._read_body()
                if not body:
                    return
                if self.path in workspace_paths:
                    self._workspace_post(self.path, body)
                    return
                if self.path == "/v1/chat/completions":
                    self._openai_completion(body)
                    return
                message = body.get("message")
                if not isinstance(message, str) or not message.strip():
                    self._send_json(400, {"error": "'message' must be a non-empty string"})
                    return
                session_id = body.get("session_id")
                if session_id is not None and (not isinstance(session_id, str) or not session_id):
                    self._send_json(400, {"error": "'session_id' must be a non-empty string or null"})
                    return
                schema = body.get("output_schema")
                if schema is not None and not isinstance(schema, dict):
                    self._send_json(400, {"error": "'output_schema' must be an object"})
                    return
                if body.get("stream") is True:
                    # ``stream: true`` returns a Server-Sent Events stream of the
                    # agent's events instead of a single JSON result body.
                    self._sse_run(message, schema, session_id)
                    return
                record = server._run_agent(
                    message,
                    stream=False,
                    output_schema=schema,
                    session_id=session_id,
                )
                self._send_json(200 if record["status"] == "completed" else 500, dict(record))

            def _openai_completion(self, body: dict[str, Any]) -> None:
                """Implement a small non-streaming OpenAI-compatible gateway.

                ``model`` intentionally identifies the Kinetic session rather
                than selecting an LLM model; model routing belongs to an
                application's agent factory / registry configuration.
                """
                model = body.get("model")
                messages = body.get("messages")
                if not isinstance(model, str) or not model:
                    self._send_json(400, {"error": "'model' must be a non-empty string"})
                    return
                if not isinstance(messages, list):
                    self._send_json(400, {"error": "'messages' must be an array"})
                    return
                message = next(
                    (item.get("content") for item in reversed(messages)
                     if isinstance(item, dict) and item.get("role") == "user"
                     and isinstance(item.get("content"), str)),
                    None,
                )
                if not isinstance(message, str) or not message.strip():
                    self._send_json(400, {"error": "'messages' must contain a non-empty user message"})
                    return
                if body.get("stream") is True:
                    self._openai_completion_stream(body, model, message)
                    return
                record = server._run_agent(message, stream=False, output_schema=None, session_id=model)
                if record["status"] != "completed":
                    self._send_json(500, dict(record))
                    return
                usage = record["usage"]
                self._send_json(200, {
                    "id": "chatcmpl-" + record["run_id"],
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": record["final"]}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": usage["input_tokens"],
                        "completion_tokens": usage["output_tokens"],
                        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                    },
                })

            def _openai_completion_stream(
                self, body: dict[str, Any], model: str, message: str
            ) -> None:
                """Stream an OpenAI-compatible SSE response for one run.

                Emits ``chat.completion.chunk`` objects as the agent publishes
                ``agent.text_delta`` events, then a final ``stop`` chunk and
                the ``[DONE]`` sentinel.
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self._response_status = 200

                run_id = uuid.uuid4().hex
                chunk_id = "chatcmpl-" + run_id
                agent = server._make_agent(model)

                def emit(delta: dict[str, Any], finish_reason: str | None) -> None:
                    chunk = json.dumps({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    })
                    self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                    self.wfile.flush()

                from kinetic_sdk.event.bus import Event

                def publish(event: Event) -> None:
                    if event.type == "agent.text_delta":
                        delta = event.payload.get("delta", "")
                        if isinstance(delta, str) and delta:
                            emit({"content": delta}, None)

                agent.event_bus.subscribe("*", publish)
                try:
                    with server._session_lock(model):
                        agent.run(message, stream=True)
                    with server._lock:
                        server._session_activity[model] = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("OpenAI streaming run failed")
                    emit({}, "stop")
                    self.wfile.write(
                        f'data: {{"error": "{type(exc).__name__}: {exc}"}}\n\n'.encode("utf-8")
                    )
                finally:
                    agent.event_bus.unsubscribe("*", publish)

                emit({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                # End the SSE response: close the connection so the client can
                # observe EOF after the [DONE] sentinel.
                self.close_connection = True

            def _websocket_run(self) -> None:
                """Upgrade then serve run requests over a persistent connection.

                The connection stays open and accepts any number of JSON text
                frames (each ``{"message": str, "session_id": str?}``), so a
                client can stream several runs without re-handshaking. Control
                frames (ping/pong/close) are handled transparently by
                :meth:`_ws_receive`; the connection ends on a close frame,
                EOF, or a protocol violation.
                """
                key = self.headers.get("Sec-WebSocket-Key")
                if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
                    self._send_json(400, {"error": "websocket upgrade required"})
                    return
                accept = base64.b64encode(hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                ).digest()).decode("ascii")
                self._response_status = 101
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                from kinetic_sdk.event.bus import Event

                while True:
                    payload = self._ws_receive()
                    if payload is None:
                        return
                    try:
                        request = json.loads(payload)
                        message = request["message"]
                        session_id = request.get("session_id")
                        if not isinstance(message, str) or not message.strip() or (session_id is not None and not isinstance(session_id, str)):
                            raise ValueError
                    except (json.JSONDecodeError, KeyError, ValueError):
                        self._ws_send(json.dumps({"error": "expected JSON with non-empty 'message'"}))
                        continue
                    agent = server._make_agent(session_id)

                    def publish(event: Event) -> None:
                        self._ws_send(json.dumps({"type": event.type, "payload": event.payload}, default=str))

                    agent.event_bus.subscribe("*", publish)
                    try:
                        if session_id is None:
                            final = agent.run(message)
                        else:
                            # Same per-session serialisation as /runs: a WS
                            # turn and a /runs turn on one cached agent must
                            # not race its non-reentrant run().
                            with server._session_lock(session_id):
                                final = agent.run(message)
                            with server._lock:
                                server._session_activity[session_id] = time.monotonic()
                        self._ws_send(json.dumps({"type": "run.completed", "final": final}))
                    except Exception as exc:  # noqa: BLE001 - never drop a connection on a run error
                        logger.exception("WebSocket run failed")
                        self._ws_send(json.dumps({"type": "run.error", "error": f"{type(exc).__name__}: {exc}"}))
                    finally:
                        agent.event_bus.unsubscribe("*", publish)

            def _ws_receive(self) -> str | None:
                """Read one complete text message, handling control frames inline.

                Returns the decoded text payload, or ``None`` when the peer
                closed the connection (close frame, EOF, or a protocol
                violation). Ping frames are answered with a pong; pong frames
                are ignored; fragmented text messages are reassembled from
                continuation frames. Data payloads are capped at
                ``MAX_WS_PAYLOAD_BYTES``.
                """
                fragments: list[bytes] = []
                while True:
                    header = self.rfile.read(2)
                    if len(header) != 2:
                        return None
                    fin = bool(header[0] & 0x80)
                    opcode = header[0] & 0x0F
                    length_byte = header[1]
                    masked = bool(length_byte & 0x80)
                    length = length_byte & 0x7F
                    if length == 126:
                        raw = self.rfile.read(2)
                        if len(raw) != 2:
                            return None
                        length = struct.unpack("!H", raw)[0]
                    elif length == 127:
                        raw = self.rfile.read(8)
                        if len(raw) != 8:
                            return None
                        length = struct.unpack("!Q", raw)[0]
                    if length > MAX_WS_PAYLOAD_BYTES:
                        self._ws_send_close(1009, "payload too large")
                        return None
                    mask = self.rfile.read(4) if masked else b""
                    if masked and len(mask) != 4:
                        return None
                    data = self.rfile.read(length)
                    if len(data) != length:
                        return None
                    if masked:
                        data = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))

                    if opcode == 0x9:  # ping -> pong
                        self._ws_send_frame(0xA, data)
                        continue
                    if opcode == 0xA:  # pong -> ignore
                        continue
                    if opcode == 0x8:  # close -> echo close, end connection
                        self._ws_send_close(1000, "")
                        return None
                    if opcode == 0x0:  # continuation frame
                        if not fragments:
                            return None  # continuation without a start frame
                        fragments.append(data)
                        if sum(len(f) for f in fragments) > MAX_WS_PAYLOAD_BYTES:
                            self._ws_send_close(1009, "payload too large")
                            return None
                        if fin:
                            return b"".join(fragments).decode("utf-8")
                        continue
                    if opcode in (0x1, 0x2):  # text/binary start frame
                        if fragments:
                            return None  # new message in the middle of a fragment
                        if fin:
                            return data.decode("utf-8")
                        fragments = [data]
                        continue
                    return None  # reserved/unknown opcode

            def _ws_send_frame(self, opcode: int, data: bytes) -> None:
                first = 0x80 | opcode
                if len(data) < 126:
                    header = bytes([first, len(data)])
                elif len(data) <= 0xFFFF:
                    header = bytes([first, 126]) + struct.pack("!H", len(data))
                else:
                    header = bytes([first, 127]) + struct.pack("!Q", len(data))
                try:
                    self.connection.sendall(header + data)
                except (BrokenPipeError, ConnectionResetError, socket.error):
                    pass

            def _ws_send(self, text: str) -> None:
                self._ws_send_frame(0x1, text.encode("utf-8"))

            def _ws_send_close(self, code: int, reason: str) -> None:
                self._ws_send_frame(0x8, struct.pack("!H", code) + reason.encode("utf-8"))

        return Handler
