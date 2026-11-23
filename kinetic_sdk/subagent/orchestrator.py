"""Task-dispatching orchestrator over a heterogeneous sub-agent catalogue.

:class:`SubagentPool` fans ONE spec out over itself (``run_all``) and
:class:`DelegateBatchTool` does the same through the agent loop. This module
is the library-level coordinator for the common manager/worker shape: a
parent hands a LIST of work-items, each explicitly routed to a NAMED
sub-agent spec (e.g. a "reviewer" agent for a review task and a "reporter"
agent for a report task), runs them concurrently under ONE shared
:class:`~kinetic_sdk.subagent.budget.SpawnBudget`, and reports per-task
outcomes in input order.

It adds what the pool deliberately leaves at a lower level:

* **Work-item routing** — every task names the spec that should run it.
* **Streaming lifecycle** — ``orchestrator.task_started`` /
  ``orchestrator.task_finished`` / ``orchestrator.task_retry`` events on an
  optional :class:`~kinetic_sdk.event.bus.EventBus`, so UIs/loggers see
  progress as workers finish, not only after the whole batch.
* **Failure isolation + bounded retry** — one broken task or spec never
  blocks the others, and ``retry_on_failure`` lets a failed task run again
  once (same spec) before giving up.

Like the rest of the sub-agent machinery, spawned agents are charged to the
shared budget and the parent's tools/policy are inherited. Delegate only for
genuinely independent work — see the module docstring of ``subagent/pool``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import run_subagent
from kinetic_sdk.subagent.manifest import SubagentSpec


@dataclass(frozen=True)
class OrchestrationTask:
    """One work item: which sub-agent should run, and on what prompt."""

    task: str
    subagent: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationResult:
    """Outcome of one dispatched work item."""

    task: OrchestrationTask
    success: bool
    final_message: str = ""
    error: str | None = None
    agent_id: str | None = None
    attempts: int = 1

    @property
    def subagent(self) -> str:
        return self.task.subagent

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.task,
            "subagent": self.task.subagent,
            "metadata": self.task.metadata,
            "success": self.success,
            "final_message": self.final_message,
            "error": self.error,
            "agent_id": self.agent_id,
            "attempts": self.attempts,
        }


class SubagentOrchestrator:
    """Route a batch of work items to named sub-agents, concurrently.

    Args:
        parent: The owning agent whose tools/policy/audit the workers inherit.
        specs: name -> :class:`SubagentSpec` catalogue. Task ``subagent``
            values must be keys of this mapping.
        max_concurrency: Bounded number of workers (and concurrency) in flight.
        event_bus: Optional bus for orchestration lifecycle events. ``None``
            disables event emission (workers still publish their own agent
            events on the shared parent bus).
    """

    def __init__(
        self,
        parent: Agent,
        specs: Mapping[str, SubagentSpec],
        *,
        max_concurrency: int = 3,
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive int")
        if not specs:
            raise ValueError("at least one sub-agent spec is required")
        self.parent = parent
        self.specs = dict(specs)
        self.max_concurrency = max_concurrency
        self.event_bus = event_bus

    def run(
        self,
        tasks: Sequence[OrchestrationTask],
        *,
        budget: SpawnBudget | None = None,
        llm_factory: Any | None = None,
        retry_on_failure: int = 0,
        timeout: float | None = None,
    ) -> list[OrchestrationResult]:
        """Dispatch *tasks* (each naming its spec) and return per-task results.

        Returns results in the caller's *tasks* order. Entries for tasks that
        did not succeed carry ``success=False`` and an ``error``; other tasks
        are unaffected (failure isolation). When ``retry_on_failure`` >= 1, a
        failed task is re-dispatched once to the SAME spec; ``attempts`` then
        reflects the number of worker invocations.

        ``budget`` (one shared :class:`SpawnBudget`) caps the whole batch's
        tool calls/tokens/cost across all workers; ``None`` creates an
        unlimited default. ``llm_factory``, when given, builds a client per
        worker (called with the spec's ``model``). ``timeout`` bounds how long
        the batch is awaited (per :func:`as_completed`).
        """
        budget = budget if budget is not None else SpawnBudget()
        for task in tasks:
            if task.subagent not in self.specs:
                raise ValueError(
                    f"unknown sub-agent {task.subagent!r} "
                    f"(registered: {sorted(self.specs)})"
                )

        base_llm_override = None
        # Workers share the parent client through a lock when none is given.
        if llm_factory is None and self.max_concurrency > 1:
            from kinetic_sdk.subagent.pool import _SerializedLLMClient

            base_llm_override = _SerializedLLMClient(self.parent.llm)

        results: dict[int, OrchestrationResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures: dict[Future[Any], int] = {}
            for index, task in enumerate(tasks):
                self._bump_started(index, task, attempt=1)
                futures[executor.submit(
                    self._run_attempt, task, budget, llm_factory, base_llm_override
                )] = index

            for future in as_completed(list(futures), timeout=timeout):
                index = futures[future]
                task = tasks[index]
                attempts, success, final, error, agent_id = future.result()
                if not success and max(retry_on_failure, 0) >= 1:
                    # One retry, same spec, then give up.
                    self._bump_retry(index, task)
                    a2, s2, f2, e2, _ = self._run_attempt(
                        task, budget, llm_factory, base_llm_override
                    )
                    attempts, success, final, error = attempts + a2, s2, f2, e2
                self._bump_finished(index, task, success=success, error=None if success else error)
                results[index] = OrchestrationResult(
                    task=task,
                    success=success,
                    final_message=final,
                    error=None if success else error,
                    agent_id=agent_id,
                    attempts=attempts,
                )

        return [results[i] for i in range(len(tasks))]

    def _run_attempt(
        self,
        task: OrchestrationTask,
        budget: SpawnBudget,
        llm_factory: Any,
        base_llm_override: Any,
    ) -> tuple[int, bool, str, str | None, str | None]:
        spec = self.specs[task.subagent]
        try:
            result = run_subagent(
                self.parent,
                spec,
                task.task,
                budget,
                llm_factory=llm_factory,
                _base_llm_override=base_llm_override,
            )
            return 1, True, result.final_message, None, result.agent_id
        except Exception as exc:  # noqa: BLE001 - isolation: one failure is not fatal
            return 1, False, "", f"{type(exc).__name__}: {exc}", None

    # -- events ---------------------------------------------------------

    def _bump_started(self, index: int, task: OrchestrationTask, *, attempt: int) -> None:
        self._emit(
            "orchestrator.task_started",
            {
                "index": index,
                "subagent": task.subagent,
                "task": task.task,
                "attempt": attempt,
            },
        )

    def _bump_retry(self, index: int, task: OrchestrationTask) -> None:
        self._emit(
            "orchestrator.task_retry",
            {"index": index, "subagent": task.subagent, "task": task.task},
        )

    def _bump_finished(
        self, index: int, task: OrchestrationTask, *, success: bool, error: str | None
    ) -> None:
        self._emit(
            "orchestrator.task_finished",
            {
                "index": index,
                "subagent": task.subagent,
                "task": task.task,
                "success": success,
                "error": error,
            },
        )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(Event(type=event_type, payload=payload, source="orchestrator"))