"""Opt-in plan/execute/verify primitives for :mod:`kinetic_sdk.agent`.

The normal agent loop deliberately remains a small, single-model-turn loop.
These interfaces add a production-friendly control plane without forcing an
extra model request (and its cost) on existing SDK users.  A planner creates
a durable plan before execution; an answer verifier can reject a purported
final answer and request a bounded correction round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Plan:
    """An immutable, human-readable execution plan.

    ``steps`` must be non-empty so an accidental empty plan never suppresses
    the model's normal reasoning. ``metadata`` is intentionally JSON-like so
    callers can attach issue IDs, workspace scope, or planner provenance.
    """

    goal: str
    steps: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("plan goal must be a non-empty string")
        if not self.steps or any(not isinstance(step, str) or not step.strip() for step in self.steps):
            raise ValueError("plan steps must contain at least one non-empty string")

    def to_prompt(self) -> str:
        """Render a non-authoritative plan for the executor's system prompt."""
        numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(self.steps, 1))
        return (
            "Execution plan (follow it when evidence supports it; update your "
            "approach if tool results contradict it):\n"
            f"Goal: {self.goal}\n{numbered}"
        )


@dataclass(frozen=True)
class VerificationResult:
    """A verdict returned after the executor proposes a final answer."""

    accepted: bool
    feedback: str = ""


class PlanStrategy(Protocol):
    """Build a plan before the agent begins its ordinary execution loop."""

    def create_plan(self, task: str, tools: list[dict[str, Any]]) -> Plan: ...


class AnswerVerifier(Protocol):
    """Check whether a final answer satisfies the task and available plan."""

    def verify(self, task: str, answer: str, plan: Plan | None) -> VerificationResult: ...


class StaticPlanStrategy:
    """Small deterministic planner useful for applications and tests.

    It is also a safe fallback for hosts that build plans outside the SDK and
    want the agent to execute exactly that plan without another LLM request.
    """

    def __init__(self, steps: list[str] | tuple[str, ...], goal: str | None = None) -> None:
        self._steps = tuple(steps)
        self._goal = goal

    def create_plan(self, task: str, tools: list[dict[str, Any]]) -> Plan:
        del tools
        return Plan(goal=self._goal or task, steps=self._steps, metadata={"source": "static"})
