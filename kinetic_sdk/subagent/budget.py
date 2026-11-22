"""Cost/behaviour safety nets for sub-agent trees (the core guardrails).

This module deliberately does not turn ordinary delegation into a pool.  Use
it only when tasks are independent (or require a substantially different
system prompt/tool set); a single agent is cheaper and simpler for a small,
serial task.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping

from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
)

#: Default total tool-call budget for one spawned tree. Sized so a healthy
#: deep delegation tree fits comfortably, while a runaway recursion (each
#: level costs >= 1 call) is stopped long before the API bill grows teeth.
DEFAULT_MAX_TOTAL_TOOL_CALLS = 4000

#: Default consecutive-identical-call limit per agent before the circuit
#: breaker trips.
DEFAULT_MAX_CONSECUTIVE_REPEATS = 20

#: Default token budget (in tokens) for one spawned tree. This is a separate
#: budget from tool calls, intended for providers that bill by token usage.
DEFAULT_MAX_TOTAL_TOKENS = 1000000  # 1M tokens default (generous but safe)


class SpawnBudget:
    """Shared tool-call budget for an entire sub-agent tree.

    ONE instance is created per root task and handed down to every spawned
    sub-agent (and to whatever those spawn in turn), so the counter reflects
    the whole tree — not one agent. Spawned agents are always charged; root
    calls are charged too when ``DelegateTool(charge_root_tool_calls=True)``
    opts in.  When it runs out the tree is throttled with :class:`BudgetExceededError`.

    Args:
        max_total_tool_calls: Hard cap on tool calls requested across the
            tree. The ``max_total_tool_calls``-th call still runs; the next
            one raises :class:`BudgetExceededError`.

        max_total_tokens: Optional hard cap on total token usage (input +
            output) across the entire tree.  When exceeded, a :class:`TokenExceededError`
            is raised.  This is separate from ``max_total_tool_calls`` so
            providers can enforce both cost and token limits independently.
    """

    def __init__(self, max_total_tool_calls: int = DEFAULT_MAX_TOTAL_TOOL_CALLS,
                 max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS) -> None:
        if not isinstance(max_total_tool_calls, int) or max_total_tool_calls < 1:
            raise ValueError(
                f"max_total_tool_calls must be a positive int, got {max_total_tool_calls!r}"
            )
        if max_total_tokens is not None and (not isinstance(max_total_tokens, int) or max_total_tokens < 0):
            raise ValueError(
                f"max_total_tokens must be a non-negative int or None, got {max_total_tokens!r}"
            )
        self._max_tool_calls = max_total_tool_calls
        self._max_tokens = max_total_tokens
        self._used_tool_calls = 0
        self._used_tokens = 0
        self._per_agent: dict[str, int] = {}
        self._lock = threading.Lock()

    def record_tool_call(self, agent_id: str, tool_name: str) -> None:
        """Charge one tool call to the shared budget.

        Raises:
            BudgetExceededError: When the budget is already exhausted. The
                counter is NOT incremented past the cap, so ``used`` never
                exceeds ``max_total_tool_calls``.
        """
        with self._lock:
            if self._used_tool_calls >= self._max_tool_calls:
                raise BudgetExceededError(
                    f"Spawn budget exhausted: {self._used_tool_calls}/{self._max_tool_calls} "
                    f"tool calls used across the sub-agent tree when agent {agent_id!r} "
                    f"requested {tool_name!r}"
                )
            self._used_tool_calls += 1
            self._per_agent[agent_id] = self._per_agent.get(agent_id, 0) + 1

    def record_token_usage(self, agent_id: str, token_count: int) -> None:
        """Charge token usage to the shared budget.

        Raises:
            TokenExceededError: When the token budget is already exhausted.
        """
        with self._lock:
            if self._max_tokens is not None and self._used_tokens + token_count > self._max_tokens:
                raise TokenExceededError(
                    f"Token budget exceeded: {self._used_tokens}/{self._max_tokens} "
                    f"tokens used across the sub-agent tree when agent {agent_id!r} "
                    f"used {token_count} tokens"
                )
            self._used_tokens += token_count

    @property
    def max_total_tool_calls(self) -> int:
        """The configured cap on tool calls."""
        return self._max_tool_calls

    @property
    def max_total_tokens(self) -> int | None:
        """The configured cap on total token usage (input + output), or None."""
        return self._max_tokens

    @property
    def max_total_tool_calls_remaining(self) -> int:
        """Calls still available before the budget trips."""
        with self._lock:
            return self._max_tool_calls - self._used_tool_calls

    @property
    def max_total_tokens_remaining(self) -> int | None:
        """Tokens still available before the token budget trips, or None."""
        with self._lock:
            return (self._max_tokens - self._used_tokens) if self._max_tokens is not None else None

    @property
    def exhausted(self) -> bool:
        """True once no further tool call may be charged."""
        with self._lock:
            return self._used_tool_calls >= self._max_tool_calls

    @property
    def exhausted_tokens(self) -> bool:
        """True once no further token usage may be charged (if a token limit is set)."""
        with self._lock:
            return self._max_tokens is not None and self._used_tokens >= self._max_tokens

    @property
    def used(self) -> int:
        """Tool calls charged so far (never exceeds the cap)."""
        with self._lock:
            return self._used_tool_calls

    @property
    def remaining(self) -> int:
        """Calls still available before the budget trips."""
        with self._lock:
            return self._max_tool_calls - self._used_tool_calls

    @property
    def remaining_tokens(self) -> int | None:
        """Tokens still available before the token budget trips, or None."""
        with self._lock:
            return (self._max_tokens - self._used_tokens) if self._max_tokens is not None else None

    @property
    def exhausted_tokens(self) -> bool:
        """True once no further token usage may be charged (if a token limit is set)."""
        with self._lock:
            return self._max_tokens is not None and self._used_tokens >= self._max_tokens

    def calls_by_agent(self) -> dict[str, int]:
        """Per-agent breakdown of charged tool calls (audit/debug aid)."""
        with self._lock:
            return dict(self._per_agent)


class TokenExceededError(RuntimeError):
    """Raised when token usage exceeds the configured token budget."""
    pass


class RepetitionCircuitBreaker:
    """Trips when ONE agent repeats the identical tool call consecutively.

    Unlike :class:`SpawnBudget` this is tracked PER AGENT — each spawned
    sub-agent gets its own breaker instance, so agent A looping does not
    consume agent B's allowance. A "repeat" means the same ``tool_name``
    AND the same arguments (compared via a stable serialisation, so dict
    key ordering is irrelevant). Any different call resets the counter.

    The ``max_consecutive_repeats``-th identical call in a row is still
    tolerated; the next one raises :class:`RepetitionLimitError`.
    """

    def __init__(
        self,
        max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS
    ) -> None:
        if not isinstance(max_consecutive_repeats, int) or max_consecutive_repeats < 1:
            raise ValueError(
                f"max_consecutive_repeats must be a positive int, got {max_consecutive_repeats!r}"
            )
        self._max = max_consecutive_repeats
        self._last_key: tuple[str, str] | None = None
        self._count = 0

    def record(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> None:
        """Register one requested tool call; raise if it trips the limit.

        Raises:
            RepetitionLimitError: When this call is identical to the
                previous one AND the consecutive-repeat count would exceed
                ``max_consecutive_repeats``.
        """
        key = (tool_name, _stable_serialise(arguments))
        if key == self._last_key:
            self._count += 1
        else:
            self._last_key = key
            self._count = 1
        if self._count > self._max:
            raise RepetitionLimitError(
                f"Repetition limit exceeded: tool {tool_name!r} called with "
                f"identical arguments {self._count} times in a row "
                f"(limit {self._max})"
            )

    @property
    def max_consecutive_repeats(self) -> int:
        """The configured consecutive-repeat cap."""
        return self._max

    @property
    def consecutive_count(self) -> int:
        """Current run length of identical consecutive calls (0 before any)."""
        return self._count

    def reset(self) -> None:
        """Forget the tracked call (the counter also self-resets on any change)."""
        self._last_key = None
        self._count = 0


def _stable_serialise(arguments: Any) -> str:
    """Serialise tool arguments so key ordering never hides a repeat."""
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(arguments)