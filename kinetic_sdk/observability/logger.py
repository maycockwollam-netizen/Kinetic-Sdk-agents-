"""Structured logging over the event bus.

The agent loop already publishes lifecycle events (``agent.run_started``,
``agent.tool_call_finished``, ``context.compacted``, ...) to an
:class:`EventBus`, but without a subscriber those events vanish when the run
ends. An :class:`ObservabilityLogger` subscribes to the wildcard ``"*"`` and
turns every event into a structured, redacted log entry so a run can be
debugged or traced afterwards.

Redaction reuses :func:`kinetic_sdk.security.redact.redact_value` — the
observability layer never implements its own scrubbing logic.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, TextIO

from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.security.redact import redact_value


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (tz-aware)."""
    return datetime.now(timezone.utc).isoformat()


class ObservabilityLogger(ABC):
    """Interface for structured event loggers.

    A logger is an event-bus subscriber: :meth:`attach` registers
    :meth:`handle` on the wildcard so every event type is captured without
    enumerating them. Each event becomes one structured entry with at least
    ``timestamp``, ``event_type``, ``run_id`` (when the emitting run set one)
    and a redacted ``payload``.
    """

    def attach(self, bus: EventBus) -> None:
        """Subscribe this logger to every event published on *bus*."""
        bus.subscribe("*", self.handle)

    def detach(self, bus: EventBus) -> None:
        """Remove a previous :meth:`attach` subscription."""
        bus.unsubscribe("*", self.handle)

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Consume one event from the bus (subscriber signature)."""

    @staticmethod
    def build_entry(event: Event) -> dict[str, Any]:
        """Build the structured, redacted log entry for *event*."""
        payload = redact_value(dict(event.payload))
        return {
            "timestamp": _utcnow_iso(),
            "event_type": event.type,
            "run_id": payload.get("run_id"),
            "payload": payload,
        }


class ConsoleObservabilityLogger(ObservabilityLogger):
    """Prints one line per event: ``[timestamp] [event_type] payload``.

    Plain, colour-free output meant for manual debugging. The payload is
    JSON-serialised; non-serialisable values fall back to ``str``.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        # Resolved at emit time (not construction) so pytest's capsys, which
        # swaps sys.stdout per test, still captures the output.
        self._stream = stream

    def handle(self, event: Event) -> None:
        entry = self.build_entry(event)
        try:
            payload = json.dumps(entry["payload"], ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            payload = str(entry["payload"])
        stream = self._stream if self._stream is not None else sys.stdout
        print(f"[{entry['timestamp']}] [{entry['event_type']}] {payload}", file=stream)


class InMemoryObservabilityLogger(ObservabilityLogger):
    """Keeps every entry in a list. Intended for tests and :class:`RunTrace`."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def handle(self, event: Event) -> None:
        self.entries.append(self.build_entry(event))

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return recorded entries, optionally filtered by event type."""
        if event_type is None:
            return list(self.entries)
        return [e for e in self.entries if e["event_type"] == event_type]
