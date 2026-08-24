"""Token usage / cost accounting for LLM calls.

Providers report per-turn usage (``input_tokens`` / ``output_tokens``, and
sometimes a computed cost) on every response. Without aggregation that data
is lost, and "how much did this run/agent burn?" becomes unanswerable — a
production blocker every mature SDK eventually solves (OpenHands logs
per-run cost; the OpenAI tracing dashboard aggregates usage).

:class:`UsageAccumulator` is a small, thread-safe running total. The agent
owns ONE instance (``agent.usage``) spanning all its runs; per-run usage is
recovered from the ``llm.usage`` events the agent emits for every LLM call
(see :class:`~kinetic_sdk.observability.trace.RunTrace.to_summary`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class UsageSnapshot:
    """Point-in-time totals, safe to serialise.

    Attributes:
        input_tokens: Sum of reported prompt tokens.
        output_tokens: Sum of reported completion tokens.
        cost_usd: Sum of provider-reported costs when available (best-effort;
            most providers report tokens only, so this stays 0.0 for them).
        calls: Number of LLM turns recorded.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form (event payloads, summaries, metrics)."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }


class UsageAccumulator:
    """Running total of LLM usage numbers, guarded by a lock.

    ``record`` accepts a provider usage mapping (as on
    :attr:`~kinetic_sdk.llm.client.LLMResponse.usage`) and ignores unknown
    keys: ``input_tokens`` / ``output_tokens`` / ``cost_usd`` are summed,
    everything else (e.g. provider-specific cache tokens) is skipped rather
    than rejected.
    """

    def __init__(self) -> None:
        self._snapshot = UsageSnapshot()
        self._lock = threading.Lock()

    def record(self, usage: Mapping[str, Any]) -> None:
        """Fold one turn's usage into the running total (empty/forged-safe)."""
        if not usage:
            return
        inp = usage.get("input_tokens")
        out = usage.get("output_tokens")
        cost = usage.get("cost_usd")
        with self._lock:
            self._snapshot.calls += 1
            if isinstance(inp, int):
                self._snapshot.input_tokens += inp
            if isinstance(out, int):
                self._snapshot.output_tokens += out
            if isinstance(cost, (int, float)):
                self._snapshot.cost_usd += float(cost)

    def snapshot(self) -> UsageSnapshot:
        """Copy of the current totals (mutating it changes nothing here)."""
        with self._lock:
            return UsageSnapshot(
                input_tokens=self._snapshot.input_tokens,
                output_tokens=self._snapshot.output_tokens,
                cost_usd=self._snapshot.cost_usd,
                calls=self._snapshot.calls,
            )
