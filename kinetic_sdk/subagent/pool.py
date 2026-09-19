"""Bounded concurrent orchestration for independent sub-agent tasks.

This module deliberately does not turn ordinary delegation into a pool.  Use
it only when tasks are independent (or require a substantially different
system prompt/tool set); a single agent is cheaper and simpler for a small,
serial task.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import DelegationResult, run_subagent
from kinetic_sdk.subagent.manifest import SubagentSpec


class SubagentResultMerger(Protocol):
    """Combine completed sub-agent results into one parent-facing message."""

    def merge(self, results: list[DelegationResult]) -> str:
        """Return the merged final message."""


class SubagentPool:
    """Run independent specs concurrently while sharing one ``SpawnBudget``.

    The budget itself serialises its counter updates, so calls made by workers
    retain the same hard total limit as sequential delegation.  Results keep
    the caller's ``specs`` order rather than completion order.
    """

    def __init__(self, max_concurrency: int = 3) -> None:
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive int")
        self.max_concurrency = max_concurrency

    def run_all(
        self, parent: Agent, specs: Sequence[SubagentSpec], budget: SpawnBudget
    ) -> list[DelegationResult]:
        """Run every spec with its description as the delegated task prompt.

        ``SubagentSpec`` deliberately contains metadata only; its
        ``description`` is therefore the natural pool task input.  Empty
        descriptions are valid and are passed through unchanged.
        """
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [
                executor.submit(run_subagent, parent, spec, spec.description, budget)
                for spec in specs
            ]
            return [future.result() for future in futures]


def _default_merge(results: list[DelegationResult]) -> str:
    """Join results without an extra model call or interpretation."""
    return "\n\n--- Sub-agent result ---\n\n".join(result.final_message for result in results)


def spawn_subagent_pool(
    parent: Agent,
    specs: Sequence[SubagentSpec],
    *,
    budget: SpawnBudget | None = None,
    merger: SubagentResultMerger | None = None,
    max_concurrency: int = 3,
) -> str:
    """Run independent agents concurrently, merge, then optionally verify.

    Delegate only for independently separable work or when a worker needs a
    meaningfully different system prompt/tool set.  Do not delegate merely to
    make a workflow appear multi-agent: one agent has less latency, cost, and
    coordination overhead when it can complete the work directly.

    When ``parent.answer_verifier`` is configured, the existing parent
    verifier is reused for the merged text.  A rejected merge raises the same
    :class:`AnswerNotVerifiedError` path as a rejected final answer.
    """
    results = SubagentPool(max_concurrency).run_all(
        parent, specs, budget if budget is not None else SpawnBudget()
    )
    merged = merger.merge(results) if merger is not None else _default_merge(results)
    if parent.answer_verifier is not None:
        accepted, feedback = parent._verify_final_answer(merged)  # noqa: SLF001
        if not accepted:
            parent._raise_answer_not_verified(merged, feedback)  # noqa: SLF001
    return merged
