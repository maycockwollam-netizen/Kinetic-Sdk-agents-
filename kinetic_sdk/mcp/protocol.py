"""JSON-RPC 2.0 message layer for MCP (Stage 4, MCP module).

MCP (Model Context Protocol) speaks JSON-RPC 2.0 over newline-delimited
JSON: every message is ONE JSON object serialised on a single line and
terminated by ``\\n`` (NOT the ``Content-Length`` header framing used by
LSP). This module defines the three message shapes, the encode/decode pair
and the request-id generator shared by the MCP client and server.

Message shapes (all carry ``"jsonrpc": "2.0"``):

* :class:`JsonRpcRequest` — ``id`` + ``method`` + optional ``params``.
  Expects a response with the same ``id``.
* :class:`JsonRpcNotification` — ``method`` + optional ``params``, NO ``id``.
  Fire-and-forget; never answered.
* :class:`JsonRpcResponse` — ``id`` + exactly one of ``result`` / ``error``
  (``error`` being ``{"code": int, "message": str, "data": ...}``).

Decode is strict on purpose: malformed JSON or a missing required field
raises :class:`MCPProtocolError` immediately instead of silently degrading
to ``None``/empty values, which would surface far away from the real bug.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Union

#: The only JSON-RPC version MCP speaks.
JSONRPC_VERSION = "2.0"


class MCPProtocolError(ValueError):
    """Raised when a wire message is malformed or misses a required field.

    The message always says WHAT was wrong (bad JSON, missing ``id``, ...)
    so the failing peer can be identified from logs alone.
    """


@dataclass
class JsonRpcRequest:
    """A JSON-RPC request: expects exactly one response with the same id."""

    id: int | str
    method: str
    params: dict[str, Any] | list[Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass
class JsonRpcNotification:
    """A JSON-RPC notification: no id, never answered."""

    method: str
    params: dict[str, Any] | list[Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass
class JsonRpcResponse:
    """A JSON-RPC response: carries either ``result`` or ``error``, never both."""

    id: int | str | None
    result: Any = None
    error: dict[str, Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    def __post_init__(self) -> None:
        if self.result is not None and self.error is not None:
            raise MCPProtocolError(
                "JsonRpcResponse cannot carry both 'result' and 'error'"
            )

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def make_error(
        cls, id: int | str | None, code: int, message: str, data: Any = None
    ) -> "JsonRpcResponse":
        """Build an error response following the JSON-RPC error object shape."""
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return cls(id=id, error=error)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            payload["error"] = self.error
        else:
            payload["result"] = self.result
        return payload


#: Any message that can travel on the wire.
JsonRpcMessage = Union[JsonRpcRequest, JsonRpcResponse, JsonRpcNotification]


def encode(message: JsonRpcMessage) -> str:
    """Serialise *message* to one JSON line terminated by ``\\n``.

    ``ensure_ascii=False`` keeps Vietnamese tool output readable on the wire
    (both sides are required to speak UTF-8 by the MCP spec).
    """
    return json.dumps(message.to_dict(), ensure_ascii=False) + "\n"


def _require(cond: bool, detail: str) -> None:
    if not cond:
        raise MCPProtocolError(f"Malformed JSON-RPC message: {detail}")


def _validate_id(value: Any) -> None:
    _require(
        value is None or (isinstance(value, (int, str)) and not isinstance(value, bool)),
        f"'id' must be an int, str or null, got {type(value).__name__}",
    )


def decode(line: str | bytes) -> JsonRpcMessage:
    """Parse one wire line into the matching message type.

    Args:
        line: A single JSON-RPC message, with or without the trailing
            newline. ``bytes`` are decoded as UTF-8.

    Raises:
        MCPProtocolError: The JSON is malformed, the top-level value is not
            an object, ``jsonrpc`` is missing/wrong, or a required field for
            the detected message type is absent or has the wrong type.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MCPProtocolError(f"Message is not valid UTF-8: {exc}") from exc
    if not isinstance(line, str):
        raise MCPProtocolError(
            f"decode() expects str/bytes, got {type(line).__name__}"
        )
    stripped = line.strip()
    if not stripped:
        raise MCPProtocolError("Empty line is not a JSON-RPC message")
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise MCPProtocolError(f"Invalid JSON: {exc}") from exc

    _require(
        isinstance(obj, dict),
        f"top-level value must be an object, got {type(obj).__name__}",
    )
    _require(
        obj.get("jsonrpc") == JSONRPC_VERSION,
        f"'jsonrpc' must be {JSONRPC_VERSION!r}, got {obj.get('jsonrpc')!r}",
    )

    has_method = "method" in obj
    has_id = "id" in obj

    if has_method:
        method = obj["method"]
        _require(
            isinstance(method, str) and method != "",
            "'method' must be a non-empty string",
        )
        params = obj.get("params")
        if has_id:
            _validate_id(obj["id"])
            return JsonRpcRequest(id=obj["id"], method=method, params=params)
        return JsonRpcNotification(method=method, params=params)

    # No method -> response. ``id`` may legitimately be None (the peer could
    # not parse our request id), but the KEY must be present.
    _require(
        has_id,
        "neither 'method' nor 'id' present - not request, response or notification",
    )
    _validate_id(obj["id"])
    has_result = "result" in obj
    has_error = "error" in obj
    _require(
        has_result != has_error,
        "a response must carry exactly one of 'result' or 'error'",
    )
    if has_error:
        error = obj["error"]
        _require(isinstance(error, dict), "'error' must be an object")
        _require(
            isinstance(error.get("code"), int) and isinstance(error.get("message"), str),
            "'error' must contain int 'code' and str 'message'",
        )
        return JsonRpcResponse(id=obj["id"], error=error)
    return JsonRpcResponse(id=obj["id"], result=obj["result"])


class RequestIdGenerator:
    """Monotonically increasing request ids, safe to share across threads.

    The agent loop is synchronous today, but ids are generated under a lock
    so a future concurrent caller cannot hand out duplicates. ``int`` ids
    are used because every MCP peer accepts them.
    """

    def __init__(self, start: int = 1) -> None:
        self._next = start
        self._lock = threading.Lock()

    def next(self) -> int:
        """Return the next id. Never repeats within one generator's lifetime."""
        with self._lock:
            value = self._next
            self._next += 1
            return value
