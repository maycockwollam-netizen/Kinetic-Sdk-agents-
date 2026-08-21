"""Audit logging for tool execution (Stage 3, part 1).

Every tool call the agent makes — allowed or denied — is recorded with its
(redacted) input, the policy's decision, and a unique id so entries can be
traced back through the event bus. Two sinks are provided: in-memory (tests,
short-lived sessions) and JSON Lines (one JSON object per line, crash-safe to
append and trivial to parse back later).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from kinetic_sdk.security.policy import PermissionDecision
from kinetic_sdk.security.redact import redact_value
from kinetic_sdk.tool.base import ToolResult


class AuditLogger:
    """Base audit logger.

    The ``log_*`` methods build a redacted entry dict and hand it to
    :meth:`_record`, which subclasses implement to choose the sink. Entry
    shape (common fields): ``id`` (unique hex), ``timestamp`` (ISO 8601),
    ``event`` (``tool_call`` / ``tool_result`` / ``permission_denied``),
    ``tool_name`` — plus event-specific fields. All free-form text is passed
    through :func:`~kinetic_sdk.security.redact.redact_value` first.
    """

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        decision: PermissionDecision,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Record a tool call and the policy decision made for it."""
        return self._entry(
            "tool_call",
            tool_name,
            timestamp,
            input=redact_value(tool_input),
            decision={
                "allowed": decision.allowed,
                "reason": decision.reason,
                "requires_confirmation": decision.requires_confirmation,
            },
        )

    def log_tool_result(
        self,
        tool_name: str,
        result: ToolResult,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Record the outcome of an executed tool call."""
        return self._entry(
            "tool_result",
            tool_name,
            timestamp,
            result={
                "is_error": result.is_error,
                "output": redact_value(result.output),
                "error": redact_value(result.error),
            },
        )

    def log_permission_denied(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        reason: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Record a call that was blocked before execution."""
        return self._entry(
            "permission_denied",
            tool_name,
            timestamp,
            input=redact_value(tool_input),
            reason=reason,
        )

    def _entry(self, event: str, tool_name: str, timestamp: datetime, **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "timestamp": timestamp.isoformat(),
            "event": event,
            "tool_name": tool_name,
        }
        entry.update(fields)
        self._record(entry)
        return entry

    def _record(self, entry: dict[str, Any]) -> None:
        raise NotImplementedError


class InMemoryAuditLogger(AuditLogger):
    """Keeps entries in a list. Intended for tests and short-lived sessions."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def _record(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


class JSONLAuditLogger(AuditLogger):
    """Appends each entry as one JSON line to a file.

    JSON Lines (rather than a JSON array) keeps every completed entry intact
    if the process crashes mid-write, and lets the file be tailed/parsed
    incrementally. Each write is flushed immediately for the same reason.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def _record(self, entry: dict[str, Any]) -> None:
        self._fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLAuditLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
