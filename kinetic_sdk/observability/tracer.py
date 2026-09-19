"""Framework-agnostic hierarchical traces built from agent events.

``Tracer`` deliberately complements, rather than changes,
:class:`OTelObservabilityLogger`.  The latter has stable, low-overhead
semantics of one OpenTelemetry run span with event annotations.  This module
turns the same event stream into real in-memory parent/child spans for callers
that need per-turn, per-tool, retry, and compaction durations.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from kinetic_sdk.event.bus import Event
from kinetic_sdk.observability.logger import ObservabilityLogger
from kinetic_sdk.security.redact import redact_value

_TERMINAL_EVENTS = {"agent.run_finished", "agent.error", "agent.cancelled"}
_EPSILON_SECONDS = 1e-9


@dataclass
class TraceSpan:
    """One timed operation in a :class:`Tracer` hierarchy.

    ``children`` is populated only on the snapshot returned by
    :meth:`Tracer.as_tree`; the flat data returned by :meth:`spans_for_run`
    is the canonical record.
    """

    span_id: str
    parent_span_id: str | None
    kind: str
    name: str
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["TraceSpan"] = field(default_factory=list)


class Tracer(ObservabilityLogger):
    """Track nested run → turn → tool → retry/compaction spans from events.

    Attach it to an :class:`~kinetic_sdk.event.bus.EventBus` just like any
    other :class:`ObservabilityLogger`.  It is optional and introduces no
    agent-loop work when not attached.  Events are redacted before becoming
    span attributes, matching the rest of the observability package.
    """

    def __init__(self) -> None:
        self._spans: dict[str, list[TraceSpan]] = {}
        self._by_id: dict[str, dict[str, TraceSpan]] = {}
        self._open_stacks: dict[str, list[str]] = {}
        self._run_spans: dict[str, str] = {}
        self._turn_spans: dict[str, str] = {}
        self._tool_spans: dict[str, dict[str, str]] = {}
        self._retry_spans: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()

    def handle(self, event: Event) -> None:
        """Consume lifecycle events and maintain their nested span boundaries."""
        payload = event.payload or {}
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        with self._lock:
            if event.type == "agent.run_started":
                self._start_run(run_id, payload)
            elif event.type == "agent.turn_started":
                self._start_turn(run_id, payload)
            elif event.type == "agent.tool_call_started":
                self._start_tool(run_id, payload)
            elif event.type == "agent.tool_retry":
                self._start_retry(run_id, payload)
            elif event.type == "agent.tool_call_finished":
                self._finish_tool(run_id, payload)
            elif event.type == "context.compacted":
                self._record_compaction(run_id, payload)
            elif event.type in _TERMINAL_EVENTS:
                self._finish_all(run_id)

    def spans_for_run(self, run_id: str) -> list[TraceSpan]:
        """Return the run's flat spans in creation order."""
        with self._lock:
            return list(self._spans.get(run_id, ()))

    def as_tree(self, run_id: str) -> TraceSpan:
        """Return a snapshot of *run_id* as a nested ``TraceSpan`` tree.

        Raises:
            KeyError: If no run span has been observed for *run_id*.
        """
        with self._lock:
            root_id = self._run_spans.get(run_id)
            records = self._by_id.get(run_id, {})
            if root_id is None or root_id not in records:
                raise KeyError(f"Unknown trace run: {run_id}")
            clones = {
                span_id: TraceSpan(
                    span_id=span.span_id,
                    parent_span_id=span.parent_span_id,
                    kind=span.kind,
                    name=span.name,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    attributes=dict(span.attributes),
                )
                for span_id, span in records.items()
            }
            for span in self._spans[run_id]:
                if span.parent_span_id is not None:
                    clones[span.parent_span_id].children.append(clones[span.span_id])
            return clones[root_id]

    def _start_run(self, run_id: str, payload: dict[str, Any]) -> None:
        # A duplicate boundary should not leave an earlier run's spans hanging.
        if run_id in self._run_spans:
            self._finish_all(run_id)
        span = self._new_span(None, "run", "kinetic.run", payload)
        self._spans[run_id] = [span]
        self._by_id[run_id] = {span.span_id: span}
        self._open_stacks[run_id] = [span.span_id]
        self._run_spans[run_id] = span.span_id
        self._tool_spans[run_id] = {}
        self._retry_spans[run_id] = {}

    def _start_turn(self, run_id: str, payload: dict[str, Any]) -> None:
        if run_id not in self._run_spans:
            return
        current = self._turn_spans.pop(run_id, None)
        if current is not None:
            self._close(run_id, current)
        span = self._append(run_id, self._run_spans[run_id], "turn", "kinetic.turn", payload)
        self._turn_spans[run_id] = span.span_id

    def _start_tool(self, run_id: str, payload: dict[str, Any]) -> None:
        if run_id not in self._run_spans:
            return
        call_id = payload.get("id")
        if not isinstance(call_id, str) or not call_id:
            return
        parent = self._turn_spans.get(run_id, self._run_spans[run_id])
        span = self._append(run_id, parent, "tool_call", "kinetic.tool_call", payload)
        self._tool_spans[run_id][call_id] = span.span_id

    def _start_retry(self, run_id: str, payload: dict[str, Any]) -> None:
        call_id = payload.get("id")
        if not isinstance(call_id, str) or call_id not in self._tool_spans.get(run_id, {}):
            return
        # Consecutive retries divide a tool span into distinct retry attempts.
        previous = self._retry_spans[run_id].get(call_id)
        if previous is not None:
            self._close(run_id, previous)
        span = self._append(
            run_id, self._tool_spans[run_id][call_id], "retry", "kinetic.tool_retry", payload
        )
        self._retry_spans[run_id][call_id] = span.span_id

    def _finish_tool(self, run_id: str, payload: dict[str, Any]) -> None:
        call_id = payload.get("id")
        if not isinstance(call_id, str):
            return
        retry_id = self._retry_spans.get(run_id, {}).pop(call_id, None)
        if retry_id is not None:
            self._close(run_id, retry_id)
        tool_id = self._tool_spans.get(run_id, {}).pop(call_id, None)
        if tool_id is not None:
            self._close(run_id, tool_id)

    def _record_compaction(self, run_id: str, payload: dict[str, Any]) -> None:
        if run_id not in self._run_spans:
            return
        parent = self._turn_spans.get(run_id, self._run_spans[run_id])
        span = self._append(run_id, parent, "compaction", "kinetic.compaction", payload)
        self._close(run_id, span.span_id)

    def _append(
        self, run_id: str, parent_span_id: str, kind: str, name: str, payload: dict[str, Any]
    ) -> TraceSpan:
        span = self._new_span(parent_span_id, kind, name, payload)
        self._spans[run_id].append(span)
        self._by_id[run_id][span.span_id] = span
        self._open_stacks[run_id].append(span.span_id)
        return span

    @staticmethod
    def _new_span(
        parent_span_id: str | None, kind: str, name: str, payload: dict[str, Any]
    ) -> TraceSpan:
        return TraceSpan(
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            start_time=time.monotonic(),
            attributes=redact_value(dict(payload)),
        )

    def _close(self, run_id: str, span_id: str) -> None:
        span = self._by_id.get(run_id, {}).get(span_id)
        if span is None or span.end_time is not None:
            return
        span.end_time = max(time.monotonic(), span.start_time + _EPSILON_SECONDS)
        stack = self._open_stacks.get(run_id, [])
        if span_id in stack:
            stack.remove(span_id)

    def _finish_all(self, run_id: str) -> None:
        for span_id in reversed(self._open_stacks.get(run_id, [])):
            self._close(run_id, span_id)
        self._turn_spans.pop(run_id, None)
        self._tool_spans.pop(run_id, None)
        self._retry_spans.pop(run_id, None)
