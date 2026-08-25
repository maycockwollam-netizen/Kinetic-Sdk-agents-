"""Opt-in cost guardrail for ROOT runs (not just sub-agent trees).

``subagent/budget.py`` ships :class:`~kinetic_sdk.subagent.budget.SpawnBudget`,
which caps tool calls across a delegation tree — but the root agent previously
had no equivalent: a looping conversation (model keeps asking for tool calls
and never produces a final answer) could burn an unbounded number of LLM
turns until ``max_iterations`` (or a provider quota) finally stopped it.

:class:`RunBudget` closes that gap with the same pattern as ``SpawnBudget``:
one counter instance handed to an :class:`~kinetic_sdk.agent.agent.Agent` /
:class:`~kinetic_sdk.agent.async_agent.AsyncAgent`, enforced at the SDK layer
(no provider billing integration), checked before every LLM call, recorded
after every response. Both limits are OPTIONAL — a budget with neither set
never blocks anything, and an agent without a budget behaves exactly as
before (fully backwards compatible with existing users).

Enforcement happens in the agent loop, which converts
:class:`RunBudgetExceeded` into a graceful stop: the event
``agent.budget_exceeded`` is emitted (with used/limit details) and the run
returns an explanatory final text — no bare exception escapes the loop.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping


class RunBudgetExceeded(Exception):
    """Raised internally when a root run exceeds its configured budget.

    The agent loop catches this before the next LLM call and converts it
    into a graceful stop, so SDK users never see it escape — it only
    surfaces when the check is invoked directly (e.g. in tests).
    """


class RunBudget:
    """Per-run cap on LLM calls and/or total tokens, enforced at the SDK.

    Args:
        max_llm_calls: Hard cap on LLM turns. The ``max_llm_calls``-th call
            still runs; the next one stops the run. ``None`` = unlimited.
        max_total_tokens: Hard cap on input+output tokens accumulated by the
            run's LLM responses. ``None`` = unlimited.

    The counter is guarded by a lock: root runs are sequential in this
    version, but the budget must already survive concurrent use for when
    parallel root execution lands (same reasoning as ``SpawnBudget``).
    """

    def __init__(
        self,
        max_llm_calls: int | None = None,
        max_total_tokens: int | None = None,
    ) -> None:
        if max_llm_calls is not None and (
            not isinstance(max_llm_calls, int) or max_llm_calls < 1
        ):
            raise ValueError(
                "max_llm_calls must be a positive int or None, "
                f"got {max_llm_calls!r}"
            )
        if max_total_tokens is not None and (
            not isinstance(max_total_tokens, int) or max_total_tokens < 1
        ):
            raise ValueError(
                "max_total_tokens must be a positive int or None, "
                f"got {max_total_tokens!r}"
            )
        self._max_calls = max_llm_calls
        self._max_tokens = max_total_tokens
        self._used_calls = 0
        self._used_tokens = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        """Verify the budget still allows another LLM call.

        Raises:
            RunBudgetExceeded: When any configured limit is already
                exhausted. The call limit is reported first when both trip.
        """
        with self._lock:
            if self._max_calls is not None and self._used_calls >= self._max_calls:
                raise RunBudgetExceeded(
                    f"LLM call budget exhausted: {self._used_calls}/"
                    f"{self._max_calls} calls used"
                )
            if self._max_tokens is not None and self._used_tokens >= self._max_tokens:
                raise RunBudgetExceeded(
                    "token budget exhausted: "
                    f"{self._used_tokens}/{self._max_tokens} tokens used "
                    "(input + output)"
                )

    def record(self, usage: Mapping[str, Any] | None) -> None:
        """Charge one completed LLM call, plus its reported token usage.

        Every call counts toward ``max_llm_calls`` even when the provider
        reports no usage numbers (otherwise a usage-silent provider would
        let a call-limited run spin). Token sums only count int values,
        mirroring :class:`~kinetic_sdk.llm.usage.UsageAccumulator`.
        """
        inp = usage.get("input_tokens") if usage else None
        out = usage.get("output_tokens") if usage else None
        with self._lock:
            self._used_calls += 1
            if isinstance(inp, int):
                self._used_tokens += inp
            if isinstance(out, int):
                self._used_tokens += out

    @property
    def max_llm_calls(self) -> int | None:
        """The configured call cap (``None`` = unlimited)."""
        return self._max_calls

    @property
    def max_total_tokens(self) -> int | None:
        """The configured token cap (``None`` = unlimited)."""
        return self._max_tokens

    @property
    def used_calls(self) -> int:
        """LLM calls recorded so far."""
        with self._lock:
            return self._used_calls

    @property
    def used_tokens(self) -> int:
        """Input + output tokens accumulated so far."""
        with self._lock:
            return self._used_tokens

    @property
    def exhausted(self) -> bool:
        """True once any configured limit has been reached."""
        with self._lock:
            if self._max_calls is not None and self._used_calls >= self._max_calls:
                return True
            if self._max_tokens is not None and self._used_tokens >= self._max_tokens:
                return True
            return False

    def details(self) -> dict[str, Any]:
        """Serialisable used/limit snapshot for ``agent.budget_exceeded``."""
        with self._lock:
            return {
                "max_llm_calls": self._max_calls,
                "max_total_tokens": self._max_tokens,
                "used_calls": self._used_calls,
                "used_tokens": self._used_tokens,
            }
