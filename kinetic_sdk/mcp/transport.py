"""Transports for MCP (Stage 4, MCP module).

A transport moves JSON-RPC messages between the SDK and an MCP peer. Two
flavours are provided:

* :class:`StdioTransport` — the peer is a local subprocess (the common case
  for MCP servers: Unity, filesystem, GitHub, ... all ship as commands the
  client spawns). Messages are newline-delimited JSON on the child's
  stdin/stdout. The same class can also *attach* to the current process's
  own stdin/stdout, which is how :class:`~kinetic_sdk.mcp.server.MCPServer`
  serves when Kinetic itself is spawned as an MCP subprocess.
* :class:`SSETransport` — the peer is remote: an SSE (Server-Sent Events)
  stream carries server->client messages, plain HTTP POSTs carry
  client->server messages.

Both transports are context managers and register a best-effort ``__del__``
cleanup so a forgotten ``close()`` (or an exception mid-handshake) does not
leave an orphaned subprocess / hanging connection.

Timeout model: ``receive(timeout=...)`` raises :class:`MCPTimeoutError`
when no complete message arrives within the deadline. Stdio reads are
pumped by a daemon reader thread into a queue, so timeouts work even
though pipes are blocking; SSE reads set the socket timeout per read.
"""

from __future__ import annotations

import http.client
import json
import queue
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Callable
from urllib.parse import urljoin, urlsplit

from kinetic_sdk.mcp.protocol import (
    JsonRpcMessage,
    MCPProtocolError,
    decode,
    encode,
)


class MCPTransportError(RuntimeError):
    """The underlying pipe/connection failed (EOF, dead subprocess, ...)."""


class MCPTimeoutError(MCPTransportError):
    """No complete message arrived within the requested deadline."""


class Transport(ABC):
    """Interface every MCP transport implements."""

    @abstractmethod
    def send(self, message: JsonRpcMessage) -> None:
        """Write one message to the peer."""

    @abstractmethod
    def receive(self, timeout: float | None = None) -> JsonRpcMessage:
        """Read the next message from the peer.

        Args:
            timeout: Seconds to wait for a complete message. ``None`` blocks
                indefinitely (appropriate on the server side).

        Raises:
            MCPTimeoutError: The deadline passed with no complete message.
            MCPTransportError: The pipe/connection died.
            MCPProtocolError: A line arrived but is not valid JSON-RPC.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the subprocess/connection. Must be idempotent."""

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------

#: Signature of the subprocess factory used by :class:`StdioTransport`.
#: Receives the argv list and the (optional) full environment, returns a
#: ``subprocess.Popen``-compatible object. Injectable so tests can substitute
#: a fake process without spawning anything.
ProcessFactory = Callable[[list[str], "dict[str, str] | None"], "subprocess.Popen[bytes]"]


def _default_process_factory(
    argv: list[str], env: dict[str, str] | None
) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        argv,
        env=env,  # None -> child inherits os.environ (Popen default)
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class _StreamEnd:
    """Sentinel the reader thread enqueues when the stream hits EOF/dies."""

    def __init__(self, detail: str) -> None:
        self.detail = detail


class StdioTransport(Transport):
    """Newline-delimited JSON-RPC over a subprocess's stdin/stdout.

    Args:
        command: Executable to spawn (e.g. ``"npx"``, ``"unity-mcp"``).
        args: Arguments for the command.
        env: Full environment for the child. ``None`` (default) inherits the
            current process environment; a dict REPLACES it entirely —
            include ``os.environ`` yourself if the child still needs it.
        process_factory: Injectable process spawner (see :data:`ProcessFactory`),
            defaults to :func:`_default_process_factory`.

    A background daemon thread pumps stdout lines into a queue so
    :meth:`receive` can enforce timeouts on blocking pipes. If the child
    dies mid-conversation (EOF on stdout), the pending and all subsequent
    :meth:`receive` calls raise :class:`MCPTransportError` carrying the
    exit code and a stderr tail — the layer above can never mistake a dead
    server for a successful no-op.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        argv = [command, *(args or [])]
        factory = process_factory or _default_process_factory
        self._proc: subprocess.Popen[bytes] | None = factory(argv, env)
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise MCPTransportError(
                "process_factory must return a Popen-like object with stdin/stdout pipes"
            )
        self._owns_process = True
        self._reader = self._proc.stdout
        self._writer = self._proc.stdin
        self._queue: queue.Queue[bytes | _StreamEnd] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._pump = threading.Thread(
            target=self._pump_lines, args=(self._reader, self._queue), daemon=True
        )
        self._pump.start()

    @classmethod
    def attach(
        cls, reader: BinaryIO | None = None, writer: BinaryIO | None = None
    ) -> "StdioTransport":
        """Attach to existing streams instead of spawning a subprocess.

        Used by the SERVER side: when Kinetic itself is launched as an MCP
        subprocess, it speaks protocol on its own stdin/stdout. Defaults to
        the current process's binary stdin/stdout. ``close()`` only flushes
        the writer — the streams belong to the host process.
        """
        self = cls.__new__(cls)
        self._proc = None
        self._owns_process = False
        self._reader = reader if reader is not None else sys.stdin.buffer
        self._writer = writer if writer is not None else sys.stdout.buffer
        self._queue = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._pump = threading.Thread(
            target=self._pump_lines, args=(self._reader, self._queue), daemon=True
        )
        self._pump.start()
        return self

    @staticmethod
    def _pump_lines(stream: BinaryIO, out: "queue.Queue[bytes | _StreamEnd]") -> None:
        try:
            while True:
                line = stream.readline()
                if line == b"":
                    out.put(_StreamEnd("EOF on stdout"))
                    return
                out.put(line)
        except Exception as exc:  # noqa: BLE001 - surfaced via _StreamEnd
            out.put(_StreamEnd(f"reader thread failed: {type(exc).__name__}: {exc}"))

    def send(self, message: JsonRpcMessage) -> None:
        if self._closed:
            raise MCPTransportError("send() on a closed StdioTransport")
        line = encode(message).encode("utf-8")
        try:
            with self._lock:
                self._writer.write(line)
                self._writer.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise MCPTransportError(
                f"Cannot write to subprocess stdin ({self._process_state()}): {exc}"
            ) from exc

    def receive(self, timeout: float | None = None) -> JsonRpcMessage:
        try:
            item = self._queue.get(timeout=timeout) if timeout is not None else self._queue.get()
        except queue.Empty:
            raise MCPTimeoutError(
                f"No message from subprocess within {timeout}s"
            ) from None
        if isinstance(item, _StreamEnd):
            # Re-enqueue so every later receive() also fails fast instead of
            # blocking forever on a dead peer.
            self._queue.put(item)
            raise MCPTransportError(self._death_detail(item))
        return decode(item)

    def _process_state(self) -> str:
        if self._proc is None:
            return "attached streams"
        code = self._proc.poll()
        return "process running" if code is None else f"process exited with code {code}"

    def _death_detail(self, end: _StreamEnd) -> str:
        parts = [f"MCP subprocess stream ended ({end.detail}); {self._process_state()}"]
        if self._proc is not None and self._proc.stderr is not None:
            try:
                tail = self._proc.stderr.read()  # safe: process already exited
            except Exception:  # noqa: BLE001 - best effort diagnostics
                tail = b""
            if tail:
                parts.append(f"stderr: {tail.decode('utf-8', 'replace')[-500:]}")
        return "; ".join(parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._owns_process or self._proc is None:
            try:
                self._writer.flush()
            except Exception:  # noqa: BLE001 - best effort
                pass
            return
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001 - child may already be gone
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    pass

    def __del__(self) -> None:  # best-effort orphan cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - __del__ must never raise
            pass


# ---------------------------------------------------------------------------
# HTTP/SSE
# ---------------------------------------------------------------------------


class SSETransport(Transport):
    """MCP over HTTP/SSE: POST requests out, SSE event stream in.

    Implements the classic MCP SSE flavour: on connect the server sends an
    ``endpoint`` event naming the URI to POST messages to; every JSON-RPC
    message the client sends is an HTTP POST to that URI (answered with a
    bare ``202``), and the server's JSON-RPC messages arrive as ``message``
    events on the long-lived GET stream.

    Args:
        url: The SSE endpoint, e.g. ``"http://localhost:8080/sse"``.
        headers: Extra HTTP headers (auth tokens, ...) sent on BOTH the SSE
            GET and every POST. Pass already-resolved plaintext here —
            credential lifecycle (``SecretValue.reveal()``) is the caller's
            job (see :mod:`kinetic_sdk.mcp.registry`).
        connect_timeout: Seconds to establish a connection and to wait for
            the initial ``endpoint`` event.
        read_timeout: Default per-read deadline on the SSE stream; can be
            overridden per :meth:`receive` call.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._conn: http.client.HTTPConnection | None = None
        self._resp: http.client.HTTPResponse | None = None
        self._post_url: str | None = None
        self._closed = False

    # -- connection lifecycle ------------------------------------------------

    def _open(self) -> None:
        if self._conn is not None:
            return
        parts = urlsplit(self._url)
        if parts.scheme not in ("http", "https"):
            raise MCPTransportError(
                f"SSETransport only supports http(s) URLs, got {self._url!r}"
            )
        conn_cls = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        conn = conn_cls(parts.hostname, parts.port, timeout=self._connect_timeout)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        try:
            conn.request(
                "GET", path, headers={"Accept": "text/event-stream", **self._headers}
            )
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise MCPTransportError(f"Cannot connect to SSE endpoint {self._url}: {exc}") from exc
        if resp.status != 200:
            conn.close()
            raise MCPTransportError(
                f"SSE endpoint {self._url} answered HTTP {resp.status}, expected 200"
            )
        self._conn = conn
        self._resp = resp
        # The spec requires the endpoint event first; honour connect_timeout
        # for it so a silent server cannot hang initialize() forever.
        event, data = self._read_event(self._connect_timeout)
        if event != "endpoint" or not data:
            self.close()
            raise MCPTransportError(
                f"SSE server did not send an 'endpoint' event first (got event={event!r})"
            )
        self._post_url = urljoin(self._url, data)

    def _read_event(self, timeout: float) -> tuple[str | None, str]:
        """Read one SSE event; returns (event_type, data)."""
        assert self._conn is not None and self._resp is not None
        sock = self._conn.sock
        if sock is not None:
            sock.settimeout(timeout)
        event: str | None = "message"  # SSE default when no 'event:' line
        data_lines: list[str] = []
        try:
            while True:
                raw = self._resp.fp.readline()  # type: ignore[union-attr]
                if raw == b"":
                    raise MCPTransportError("SSE stream closed by the server")
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":  # blank line dispatches the event
                    if data_lines:
                        return event, "\n".join(data_lines)
                    continue
                if line.startswith(":"):  # comment / heartbeat
                    continue
                field, _, value = line.partition(":")
                value = value.lstrip(" ")
                if field == "event":
                    event = value
                elif field == "data":
                    data_lines.append(value)
        except TimeoutError as exc:
            raise MCPTimeoutError(
                f"No SSE event within {timeout}s from {self._url}"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise MCPTransportError(f"SSE stream from {self._url} failed: {exc}") from exc

    # -- Transport interface ---------------------------------------------------

    def send(self, message: JsonRpcMessage) -> None:
        if self._closed:
            raise MCPTransportError("send() on a closed SSETransport")
        self._open()
        assert self._post_url is not None
        parts = urlsplit(self._post_url)
        conn_cls = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        conn = conn_cls(parts.hostname, parts.port, timeout=self._connect_timeout)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        body = encode(message).encode("utf-8")
        try:
            conn.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    **self._headers,
                },
            )
            resp = conn.getresponse()
            resp.read()  # drain so the connection can close cleanly
        except (OSError, http.client.HTTPException) as exc:
            raise MCPTransportError(f"POST to MCP message endpoint failed: {exc}") from exc
        finally:
            conn.close()
        if resp.status not in (200, 202):
            raise MCPTransportError(
                f"MCP message endpoint answered HTTP {resp.status}, expected 200/202"
            )

    def receive(self, timeout: float | None = None) -> JsonRpcMessage:
        if self._closed:
            raise MCPTransportError("receive() on a closed SSETransport")
        self._open()
        deadline = timeout if timeout is not None else self._read_timeout
        while True:
            event, data = self._read_event(deadline)
            if event == "endpoint":  # re-announced endpoint; adopt it
                self._post_url = urljoin(self._url, data)
                continue
            if event != "message":
                continue  # unknown event types are ignored per SSE rules
            return decode(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._resp = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._conn = None

    def __del__(self) -> None:  # best-effort orphan cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - __del__ must never raise
            pass
