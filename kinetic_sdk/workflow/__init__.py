"""A thin, declarative composition layer over the agent + sub-agent primitives.

Instead of hand-wiring ``run_subagent``, ``map_reduce_subagents``,
``produce_and_review`` and ``delegate_naturally`` together, :class:`Workflow`
lets you describe a PIPELINE of steps once and ``run()`` it: each step is a
named, reusable operation sharing ONE ``SpawnBudget``, and every step's output
message is captured so a server/CLI/test can inspect per-step results.

Steps are run SEQUENTIALLY (fits the "controller -> workers" shape). If you
need fan-out, use :meth:`Workflow.map_reduce` (parallel items inside one step)
or the pool/orchestrator directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import run_subagent
from kinetic_sdk.subagent.manifest import SubagentSpec
from kinetic_sdk.subagent.pool import (
    SubagentResultMerger,
    map_reduce_subagents,
)
from kinetic_sdk.subagent.recruiter import delegate_naturally
from kinetic_sdk.subagent.review import produce_and_review

#: One workflow step: ``(parent, budget) -> output message``.
StepFn = Callable[[Agent, SpawnBudget], str]


@dataclass(frozen=True)
class WorkflowResult:
    """Outcome of executing a :class:`Workflow` (one entry per step)."""

    final_message: str
    outputs: dict[str, str] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"final_message": self.final_message, "outputs": self.outputs, "usage": self.usage}


class Workflow:
    """Build a sequential pipeline of named steps, then :meth:`run` it.

    ``budget`` is SHARED across all steps (the tree `SpawnBudget`), so total
    tool calls/tokens/cost stay bounded for the whole workflow. ``llm_factory``
    is forwarded to every sub-agent spawn (one client per worker when given).
    """

    def __init__(
        self,
        parent: Agent,
        *,
        budget: SpawnBudget | None = None,
        llm_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._parent = parent
        self._budget = budget if budget is not None else SpawnBudget()
        self._llm_factory = llm_factory
        self._steps: list[tuple[str, StepFn]] = []

    # -- generic ----------------------------------------------------------

    def step(self, name: str, fn: StepFn) -> "Workflow":
        """Register a raw step (*fn* receives ``(parent, budget)`` -> message)."""
        if not name or not isinstance(name, str):
            raise ValueError("step name must be a non-empty string")
        if any(existing == name for existing, _ in self._steps):
            raise ValueError(f"duplicate step name {name!r}")
        self._steps.append((name, fn))
        return self

    # -- built-in steps ----------------------------------------------------

    def run_agent(self, name: str, message: str) -> "Workflow":
        """Run a plain :meth:`Agent.run` turn as a step."""
        return self.step(name, lambda parent, _budget: parent.run(message))

    def delegate(self, name: str, spec: SubagentSpec, task_prompt: str) -> "Workflow":
        """Delegate one task to *spec* (a single sub-agent step)."""
        factory = self._llm_factory

        def _run(parent: Agent, budget: SpawnBudget) -> str:
            return run_subagent(parent, spec, task_prompt, budget, llm_factory=factory).final_message

        return self.step(name, _run)

    def map_reduce(
        self,
        name: str,
        spec: SubagentSpec,
        items: Sequence[Any],
        *,
        prompt_for: Callable[[Any, int], str] | None = None,
        merger: SubagentResultMerger | None = None,
        max_concurrency: int = 3,
    ) -> "Workflow":
        """Run *spec* over *items* in parallel and merge (one step)."""
        factory = self._llm_factory

        def _run(parent: Agent, budget: SpawnBudget) -> str:
            return map_reduce_subagents(
                parent,
                spec,
                items,
                prompt_for=prompt_for,
                merger=merger,
                budget=budget,
                max_concurrency=max_concurrency,
                llm_factory=factory,
            ).merged

        return self.step(name, _run)

    def produce_review(
        self,
        name: str,
        producer: SubagentSpec,
        critic: SubagentSpec,
        *,
        task_prompt: str,
        criteria: str = "",
        max_revisions: int = 2,
    ) -> "Workflow":
        """Producer drafts, critic reviews, revise to acceptance (one step)."""
        factory = self._llm_factory

        def _run(parent: Agent, budget: SpawnBudget) -> str:
            return produce_and_review(
                parent,
                producer,
                critic,
                task_prompt,
                criteria=criteria,
                budget=budget,
                max_revisions=max_revisions,
                llm_factory=factory,
            ).result

        return self.step(name, _run)

    def route(
        self,
        name: str,
        specs: Mapping[str, SubagentSpec],
        task: str,
    ) -> "Workflow":
        """Let a router pick the best spec for *task*, then run it (one step)."""
        budget = self._budget
        factory = self._llm_factory

        def _run(parent: Agent, _budget: SpawnBudget) -> str:
            return delegate_naturally(
                parent, specs, task, budget=budget, llm_factory=factory
            ).final_message

        return self.step(name, _run)

    # -- execution ----------------------------------------------------------

    def run(self) -> WorkflowResult:
        """Execute every step in order, capturing each output message."""
        outputs: dict[str, str] = {}
        final_message = ""
        for name, fn in self._steps:
            final_message = fn(self._parent, self._budget)
            outputs[name] = final_message
        return WorkflowResult(
            final_message=final_message,
            outputs=outputs,
            usage=self._parent.usage.snapshot().to_dict(),
        )