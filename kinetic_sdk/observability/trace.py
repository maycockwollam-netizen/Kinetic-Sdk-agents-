"""Per-run tracing built on top of structured log entries.

Every :meth:`Agent.run` call stamps its events with a unique ``run_id``.
:class:`RunTrace` collects the entries of one such run (filtered by that id)
and answers the questions asked most often when debugging a run: how long it
took, which tools it called, how the FLASH/MAX mode evolved, and whether
anything notable (escalation, permission denial, compaction) happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp produced by the observability logger."""
    return datetime.fromisoformat(value)


@dataclass
class RunTrace:
    """The recorded events of a single ``Agent.run`` invocation.

    Attributes:
        run_id: The UUID the run stamped onto all of its events.
        events: Structured log entries (as produced by
            :class:`InMemoryObservabilityLogger`) belonging to this run, in
            the order they were recorded.
    """

    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def collect(cls, entries: Iterable[dict[str, Any]], run_id: str) -> "RunTrace":
        """Build a trace from *entries* (e.g. ``logger.entries``) for *run_id*."""
        return cls(run_id=run_id, events=[e for e in entries if e.get("run_id") == run_id])

    def _of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == event_type]

    def duration(self) -> float:
        """Seconds between ``agent.run_started`` and ``agent.run_finished``.

        Returns ``0.0`` when either boundary event is missing (e.g. the run
        raised before finishing).
        """
        started = self._of_type("agent.run_started")
        finished = self._of_type("agent.run_finished")
        if not started or not finished:
            return 0.0
        start = _parse_ts(started[0]["timestamp"])
        end = _parse_ts(finished[-1]["timestamp"])
        return (end - start).total_seconds()

    def tool_calls(self) -> list[dict[str, Any]]:
        """One entry per finished tool call: name, id and success flag."""
        return [
            {
                "name": e["payload"].get("name"),
                "id": e["payload"].get("id"),
                "is_error": e["payload"].get("is_error", False),
            }
            for e in self._of_type("agent.tool_call_finished")
        ]

    def mode_transitions(self) -> list[dict[str, Any]]:
        """Chronological mode history: initial classification + escalations."""
        transitions: list[dict[str, Any]] = []
        for e in self.events:
            if e.get("event_type") == "agent.classified":
                transitions.append(
                    {
                        "event": "classified",
                        "mode": e["payload"].get("mode"),
                        "complexity": e["payload"].get("complexity"),
                        "timestamp": e["timestamp"],
                    }
                )
            elif e.get("event_type") == "agent.escalated":
                transitions.append(
                    {
                        "event": "escalated",
                        "from": e["payload"].get("from"),
                        "to": e["payload"].get("to"),
                        "timestamp": e["timestamp"],
                    }
                )
        return transitions

    def final_mode(self) -> str | None:
        """The mode the run ended in (from ``run_finished``, else transitions)."""
        finished = self._of_type("agent.run_finished")
        if finished:
            return finished[-1]["payload"].get("mode")
        transitions = self.mode_transitions()
        if not transitions:
            return None
        last = transitions[-1]
        return last.get("to") if last["event"] == "escalated" else last.get("mode")

    def to_summary(self) -> dict[str, Any]:
        """Compact one-run summary for quick inspection or aggregation later."""
        calls = self.tool_calls()
        return {
            "run_id": self.run_id,
            "duration": self.duration(),
            "tool_call_count": len(calls),
            "tool_calls_failed": sum(1 for c in calls if c["is_error"]),
            "final_mode": self.final_mode(),
            "escalated": bool(self._of_type("agent.escalated")),
            "permission_denied": bool(self._of_type("security.permission_denied")),
            "context_compacted": bool(self._of_type("context.compacted")),
        }
