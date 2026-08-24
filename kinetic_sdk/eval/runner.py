"""Eval harness: score an agent against a dataset of cases.

Mocked tests answer "does the loop work"; evals answer "does the agent
BEHAVE well" (Pydantic Evals, AutoGenBench, ADK eval all exist because unit
tests can't measure behaviour). The runner here stays minimal and fully
local:

* cases are ``EvalCase(input, expected=?, tags=[...])``,
* scorers are callables ``(case, trace, output) -> EvalScore`` — built-ins
  cover exact-match / substring / tool-was-called / no-failures,
* the runner executes cases SEQUENTIALLY (deterministic order, easy to
  debug) and builds a fresh agent per case via the injected factory, so
  state never leaks between cases,
* results are plain dicts — pipe them into your own dashboards.

The runner never calls the network itself: the factory decides whether the
cases run against a ``MockLLMClient`` (CI) or a real provider (offline
grading runs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.observability.logger import InMemoryObservabilityLogger
from kinetic_sdk.observability.trace import RunTrace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCase:
    """One input plus optional expectation.

    Attributes:
        input: The user message handed to the agent.
        expected: Reference answer (scorer-specific semantics).
        tags: Free-form grouping labels (e.g. ``["bugfix", "hard"]``).
    """

    input: str
    expected: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalScore:
    """One scorer's verdict on one case."""

    name: str
    passed: bool
    value: float = 0.0
    details: str = ""


class EvalScorer(Protocol):
    """Callable scorer: ``(case, trace, output) -> EvalScore``."""

    def __call__(
        self, case: EvalCase, trace: RunTrace, output: str
    ) -> EvalScore: ...


# --- built-in scorers -----------------------------------------------------


def exact_match(case: EvalCase, trace: RunTrace, output: str) -> EvalScore:
    """Output must equal the expected answer after stripping."""
    expected = (case.expected or "").strip()
    ok = output.strip() == expected
    return EvalScore("exact_match", ok, 1.0 if ok else 0.0, f"expected {expected!r}")


def contains_expected(case: EvalCase, trace: RunTrace, output: str) -> EvalScore:
    """Output must contain the expected substring."""
    expected = case.expected or ""
    ok = expected in output
    return EvalScore("contains_expected", ok, 1.0 if ok else 0.0, f"needle {expected!r}")


def no_tool_failures(case: EvalCase, trace: RunTrace, output: str) -> EvalScore:
    """Every executed tool call must have succeeded."""
    summary = trace.to_summary()
    failed = summary["tool_calls_failed"]
    return EvalScore(
        "no_tool_failures", failed == 0, 1.0 if failed == 0 else 0.0, f"{failed} failures"
    )


def tool_called(name: str) -> EvalScorer:
    """Factory: passes when the agent called the given tool at least once."""

    def scorer(case: EvalCase, trace: RunTrace, output: str) -> EvalScore:
        hits = sum(1 for c in trace.tool_calls() if c["name"] == name)
        return EvalScore(
            f"tool_called[{name}]", hits > 0, float(hits), f"{hits} call(s)"
        )

    return scorer


# --- runner ---------------------------------------------------------------


@dataclass
class CaseResult:
    """Everything the runner knows about one executed case."""

    case: EvalCase
    output: str
    run_id: str | None
    error: str | None
    scores: list[EvalScore]

    @property
    def passed(self) -> bool:
        """All scorers passed and the run did not crash."""
        return self.error is None and all(s.passed for s in self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form (reports, CI artefacts)."""
        return {
            "input": self.case.input,
            "expected": self.case.expected,
            "tags": list(self.case.tags),
            "output": self.output,
            "run_id": self.run_id,
            "error": self.error,
            "passed": self.passed,
            "scores": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "value": s.value,
                    "details": s.details,
                }
                for s in self.scores
            ],
        }


class EvalRunner:
    """Run a dataset of cases against freshly built agents.

    Args:
        agent_factory: Zero-arg callable returning a ready agent (each case
            gets its OWN agent — cross-case state leakage is the classic
            way to get evals that pass locally and fail in CI).
        scorers: Scorers applied to every case. Default: ``contains_expected``
            for cases with an expected value plus ``no_tool_failures``.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        scorers: Iterable[EvalScorer] | None = None,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self._agent_factory = agent_factory
        self._scorers = list(scorers) if scorers is not None else None

    def run(self, cases: Iterable[EvalCase]) -> dict[str, Any]:
        """Execute every case and return the aggregate report."""
        results: list[CaseResult] = []
        for case in cases:
            results.append(self._run_one(case))
        passed = sum(1 for r in results if r.passed)
        return {
            "total": len(results),
            "passed": passed,
            "pass_rate": passed / len(results) if results else 0.0,
            "results": [r.to_dict() for r in results],
        }

    def _run_one(self, case: EvalCase) -> CaseResult:
        obs = InMemoryObservabilityLogger()
        try:
            agent = self._agent_factory()
        except Exception as exc:  # noqa: BLE001 - bad factory is a case error
            return CaseResult(case, "", None, f"agent_factory: {exc}", [])
        # Route the run's events into the per-case trace store (the factory
        # built the agent with its own bus; attach our logger there).
        obs.attach(agent.event_bus)
        try:
            output = agent.run(case.input)
        except Exception as exc:  # noqa: BLE001 - the eval reports, not crashes
            logger.exception("Case crashed: %r", case.input)
            return CaseResult(case, "", agent.run_id, f"{type(exc).__name__}: {exc}", [])
        run_id = agent.run_id
        trace = RunTrace.collect(obs.entries, run_id) if run_id else RunTrace("", [])
        scorers = self._scorers if self._scorers is not None else (
            ([contains_expected] if case.expected is not None else []) + [no_tool_failures]
        )
        scores = [scorer(case, trace, output) for scorer in scorers]
        return CaseResult(case, output, run_id, None, scores)
