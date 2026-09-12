"""Metrics aggregation over the event stream.

Events answer "what happened"; metric snapshots answer "how often / how
much / how long". ``MetricsCollector`` subscribes to the wildcard (same
contract as :class:`~kinetic_sdk.observability.logger.ObservabilityLogger`)
and keeps counters/timers cheap enough to always stay attached:

* counts per event type (``agent.escalated``, ``hooks.error``, ...),
* tool calls vs failures (rolling totals),
* token usage totals (from ``llm.usage`` events),
* finished-run durations (from run_started/run_finished pairing),
* concurrent run spans (handles sub-agent trees sharing one bus).

The snapshot is a plain dict so callers can ship it to StatsD/Prometheus —
the SDK deliberately ships no metrics backend (optional-dependency rule).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from kinetic_sdk.event.bus import Event
from kinetic_sdk.observability.logger import ObservabilityLogger


class MetricsCollector(ObservabilityLogger):
    """Count events, tools, usage and run durations over a bus."""

    def __init__(self) -> None:
        self.event_counts: dict[str, int] = {}
        self.tool_calls = 0
        self.tool_failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.runs_started = 0
        self.runs_finished = 0
        self.run_durations: dict[str, float] = {}
        self._open_runs: dict[str, str] = {}
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        with self._lock:
            event_type = event.type
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
            payload = event.payload or {}
            run_id = payload.get("run_id")
            if event_type == "agent.run_started" and run_id:
                self.runs_started += 1
                self._open_runs[run_id] = datetime.now(timezone.utc).isoformat()
                return
            if event_type == "agent.run_finished" and run_id:
                self.runs_finished += 1
                started = self._open_runs.pop(run_id, None)
                if started is not None:
                    elapsed = (
                        datetime.now(timezone.utc) - datetime.fromisoformat(started)
                    ).total_seconds()
                    self.run_durations[run_id] = elapsed
                return
            if event_type == "agent.tool_call_finished":
                self.tool_calls += 1
                if payload.get("is_error"):
                    self.tool_failures += 1
                return
            if event_type == "llm.usage":
                usage = payload.get("usage") or {}
                if isinstance(usage.get("input_tokens"), int):
                    self.input_tokens += usage["input_tokens"]
                if isinstance(usage.get("output_tokens"), int):
                    self.output_tokens += usage["output_tokens"]
                if isinstance(usage.get("cost_usd"), (int, float)):
                    self.cost_usd += usage["cost_usd"]
                return

    def snapshot(self) -> dict[str, Any]:
        """Current counters as a plain dict (safe to serialise)."""
        with self._lock:
            durations = list(self.run_durations.values())
            return {
                "event_counts": dict(sorted(self.event_counts.items())),
                "tool_calls": self.tool_calls,
                "tool_failures": self.tool_failures,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "cost_usd": round(self.cost_usd, 6),
                },
                "runs_started": self.runs_started,
                "runs_finished": self.runs_finished,
                "open_runs": list(sorted(self._open_runs)),
                "run_duration_avg": (
                    sum(durations) / len(durations) if durations else 0.0
                ),
                "run_duration_max": max(durations) if durations else 0.0,
            }
