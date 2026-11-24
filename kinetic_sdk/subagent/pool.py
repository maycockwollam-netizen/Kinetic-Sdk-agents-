"""Bounded concurrent orchestration for independent sub-agent tasks.

This module deliberately does not turn ordinary delegation into a pool.  Use
it only when tasks are independent (or require a substantially different
system prompt/tool set); a single agent is cheaper and simpler for a small,
serial task.
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.llm.client import LLMClient, LLMResponse, Message, StreamEvent
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import (
    DelegationResult,
    SubagentStatus,
    run_subagent,
)
from kinetic_sdk.subagent.manifest import SubagentSpec


class SubagentResultMerger(Protocol):
    """Combine completed sub-agent results into one parent-facing message."""

    def merge(self, results: Sequence[DelegationResult | None]) -> str:
        """Return the merged final message."""


class _SerializedLLMClient(LLMClient):
    """Serialize calls to a shared client that pool workers inherit.

    A pool cannot assume a provider client is thread-safe.  This wrapper is
    deliberately shared by all workers, so its single lock protects the
    underlying client rather than merely protecting each worker's guard.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.model = getattr(inner, "model", "unknown")

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        with self._lock:
            return self._inner.chat(messages, tools=tools, system=system, **kwargs)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        # Hold the lock for the complete stream: releasing it after creating
        # an iterator would allow two workers to drive a stateful stream at
        # the same time.
        with self._lock:
            yield from self._inner.chat_stream(messages, tools=tools, system=system, **kwargs)


class SubagentPool:
    """Run independent specs concurrently while sharing one ``SpawnBudget``.

    The budget itself serialises its counter updates, so calls made by workers
    retain the same hard total limit as sequential delegation.  Results keep
    the caller's ``specs`` order rather than completion order.  If no
    ``llm_factory`` is supplied and multiple workers run concurrently, the
    inherited parent client is shared through an internal lock; LLM calls are
    therefore safe but serialised.  Pass ``llm_factory`` to give each worker
    its own client and retain concurrent LLM calls.
    """

    def __init__(self, max_concurrency: int = 3) -> None:
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive int")
        self.max_concurrency = max_concurrency

    def run_all(
        self,
        parent: Agent,
        specs: Sequence[SubagentSpec],
        budget: SpawnBudget,
        *,
        llm_factory: Any | None = None,
        timeout: float | None = None,
    ) -> list[DelegationResult | None]:
        """Run every spec with its description as the delegated task prompt.

        Returns one entry per spec, preserving ``specs`` order.  Entries are
        ``DelegationResult`` for successful workers and ``None`` for workers
        that raised an exception (the exception itself is swallowed so the
        pool can keep collecting other results).

        ``timeout``, when set, bounds how long the whole batch is awaited
        (passed to :func:`as_completed`). On expiry a :class:`TimeoutError`
        is raised; already-running workers are not interrupted — the
        executor waits for them on shutdown.
        """
        base_llm_override = (
            _SerializedLLMClient(parent.llm)
            if llm_factory is None and self.max_concurrency > 1
            else None
        )

        results: list[DelegationResult | None] = [None] * len(specs)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_idx: dict[Future[DelegationResult], int] = {}
            for i, spec in enumerate(specs):
                fut = executor.submit(
                    run_subagent,
                    parent,
                    spec,
                    spec.description,
                    budget,
                    llm_factory=llm_factory,
                    _base_llm_override=base_llm_override,
                )
                future_to_idx[fut] = i

            for future in as_completed(future_to_idx.keys(), timeout=timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:  # noqa: BLE001 - swallow so pool keeps going
                    results[idx] = None

        return results


class _DefaultMerge:
    """Default :class:`SubagentResultMerger`: join results verbatim."""

    def merge(self, results: Sequence[DelegationResult | None]) -> str:
        return "\n\n--- Sub-agent result ---\n\n".join(
            result.final_message for result in results if result is not None
        )


_default_merge = _DefaultMerge()


def spawn_subagent_pool(
    parent: Agent,
    specs: Sequence[SubagentSpec],
    *,
    budget: SpawnBudget | None = None,
    merger: SubagentResultMerger | None = None,
    max_concurrency: int = 3,
    llm_factory: Any | None = None,
    timeout: float | None = None,
) -> str:
    """Run independent agents concurrently, merge, then optionally verify.

    Delegate only for independently separable work or when a worker needs a
    meaningfully different system prompt/tool set.  Do not delegate merely to
    make a workflow appear multi-agent: one agent has less latency, cost, and
    coordination overhead when it can complete the work directly.

    Without ``llm_factory``, concurrent workers serialize calls to the shared
    parent LLM client for thread safety. Supplying a factory creates a client
    per worker, including workers that inherit the parent's model. When
    ``parent.answer_verifier`` is configured, the existing parent
    verifier is reused for the merged text.  A rejected merge raises the same
    :class:`AnswerNotVerifiedError` path as a rejected final answer.
    """
    budget = budget or SpawnBudget()
    pool = SubagentPool(max_concurrency=max_concurrency)
    results = pool.run_all(
        parent,
        specs,
        budget,
        llm_factory=llm_factory,
        timeout=timeout,
    )

    merger = merger or _default_merge
    merged = merger.merge(results) if results else ""

    if parent.answer_verifier is not None:
        accepted, feedback = parent._verify_final_answer(merged)  # noqa: SLF001
        if not accepted:
            parent._raise_answer_not_verified(merged, feedback)  # noqa: SLF001

    return merged


# ---------------------------------------------------------------------------
# Map-reduce: run ONE spec over a list of items in parallel, then collapse.
# ---------------------------------------------------------------------------


def _result_vote_value(result: DelegationResult) -> str:
    """A stable atom to vote on from a structured sub-agent result.

    Prefers the structured ``data`` (a bare string, or a ``choice`` field of a
    dict); falls back to the final message. Lets a MajorityVoteMerger work on
    structured handoffs (from ``SubagentSpec.result_schema``).
    """
    data = result.data
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        choice = data.get("choice")
        if isinstance(choice, str):
            return choice
        if "answer" in data and isinstance(data["answer"], str):
            return data["answer"]
    return result.final_message


class FirstSuccessMerger:
    """Return the FIRST succeeded result; otherwise the first non-None one.

    Useful when the item generous redundancy (several workers answer the same
    question) or when a single authoritative answer is enough.
    """

    def merge(self, results: Sequence[DelegationResult | None]) -> str:
        first = next(
            (r for r in results if r is not None and r.status is SubagentStatus.SUCCESS),
            None,
        )
        if first is None:
            first = next((r for r in results if r is not None), None)
        return first.final_message if first is not None else ""


class MajorityVoteMerger:
    """Pick the most common value across structured results (ties -> earliest).

    Votes come from :func:`_result_vote_value`. Use it to run several
    independent sub-agents on the SAME question and take a consensus.
    """

    def merge(self, results: Sequence[DelegationResult | None]) -> str:
        counts: Counter[str] = Counter()
        order: dict[str, int] = {}
        for rank, result in enumerate(results):
            if result is None:
                continue
            value = _result_vote_value(result)
            counts[value] += 1
            order.setdefault(value, rank)
        if not counts:
            return ""
        winner = max(counts, key=lambda value: (counts[value], -order[value]))
        votes = counts[winner]
        suffix = "s" if votes != 1 else ""
        return f"{winner} ({votes} vote{suffix})"


#: Map item (index) -> task prompt for a sub-agent.
MapPromptFor = Callable[[Any, int], str]


@dataclass
class MapReduceOutcome:
    """Result of :func:`map_reduce_subagents` (merged text + per-item results)."""

    merged: str
    results: list[DelegationResult | None]

    def __len__(self) -> int:
        return len(self.results)


def map_reduce_subagents(
    parent: Agent,
    spec: SubagentSpec,
    items: Sequence[Any],
    *,
    prompt_for: MapPromptFor | None = None,
    merger: SubagentResultMerger | None = None,
    budget: SpawnBudget | None = None,
    max_concurrency: int = 3,
    llm_factory: Any | None = None,
    timeout: float | None = None,
) -> MapReduceOutcome:
    """Run *spec* concurrently over *items* (map) and collapse the results (reduce).

    Each item is turned into a ``task_prompt`` via ``prompt_for(item, index)``
    (default ``str(item)``) and dispatched to its own sub-agent worker sharing
    one ``SpawnBudget``. The per-item :class:`DelegationResult`s (order
    preserved; ``None`` for a worker that raised) are then collapsed by
    ``merger`` (default: a verbatim join, ``FirstSuccessMerger`` picks the
    first success, ``MajorityVoteMerger`` takes a consensus of `data`).

    Only for genuinely independent items — a single agent is cheaper for a
    small serial task.
    """
    budget = budget or SpawnBudget()
    base_override = (
        _SerializedLLMClient(parent.llm)
        if llm_factory is None and max_concurrency > 1
        else None
    )
    results: list[DelegationResult | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[DelegationResult], int] = {}
        for index, item in enumerate(items):
            task_prompt = prompt_for(item, index) if prompt_for is not None else str(item)
            futures[executor.submit(
                run_subagent,
                parent,
                spec,
                task_prompt,
                budget,
                llm_factory=llm_factory,
                _base_llm_override=base_override,
            )] = index
        for future in as_completed(list(futures), timeout=timeout):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception:  # noqa: BLE001 - isolate one failing worker
                results[index] = None

    merger = merger or _default_merge
    return MapReduceOutcome(merged=merger.merge(results), results=results)