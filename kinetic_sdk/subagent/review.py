"""Produce-then-review: a producer sub-agent + a critic, iterating to acceptance.

A common two-role pattern: a *producer* sub-agent drafts an answer for a task,
a separate *critic* sub-agent checks it against criteria, and — when the
critic rejects it — the producer is re-run with the critic's feedback, up to a
bounded number of revisions. Keeps roles strictly separate (the critic never
rewrites; the producer never self-grades) with the same isolation guarantees
as any delegation: only final messages flow back, everything shares one
``SpawnBudget``.

A critic without a ``result_schema`` is treated conservatively (its review is
never treated as approval), so give the critic a schema with a boolean
``approved`` (and optionally ``feedback``) to enable auto-accept loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import DelegationResult, run_subagent
from kinetic_sdk.subagent.manifest import SubagentSpec


@dataclass(frozen=True)
class ProduceReviewResult:
    """Outcome of a produce-then-review loop.

    Attributes:
        result: The accepted (or last) producer answer.
        approved: Whether the critic approved it (``False`` when revisions
            were exhausted or the critic had no approval signal).
        revisions: Number of rejected -> revise rounds actually executed
            (``0`` means the first production was approved).
        producer: The final producer run.
        reviews: Every critic verdict, in order.
    """

    result: str
    approved: bool
    revisions: int
    producer: DelegationResult
    reviews: list[DelegationResult]

    @property
    def last_review_feedback(self) -> str:
        """The most recent critic feedback (empty for a fresh review)."""
        if not self.reviews:
            return ""
        review = self.reviews[-1]
        if isinstance(review.data, dict) and isinstance(review.data.get("feedback"), str):
            return review.data["feedback"]
        return review.final_message


def _critic_verdict(review: DelegationResult) -> tuple[bool, str]:
    """Extract ``(approved, feedback)`` from a critic result (conservative).

    Without a trustworthy structured ``approved`` flag the review is never
    treated as approval.
    """
    data = review.data
    if isinstance(data, dict) and isinstance(data.get("approved"), bool):
        feedback = data.get("feedback")
        return data["approved"], feedback if isinstance(feedback, str) else ""
    return False, review.final_message


def produce_and_review(
    parent: Agent,
    producer_spec: SubagentSpec,
    critic_spec: SubagentSpec,
    task_prompt: str,
    *,
    criteria: str = "",
    budget: SpawnBudget | None = None,
    max_revisions: int = 2,
    llm_factory: Callable[[str], Any] | None = None,
) -> ProduceReviewResult:
    """Draft with *producer_spec*, review with *critic_spec*, revise to acceptance.

    The producer's first answer is handed to the critic; if the critic
    signals non-approval, the producer is re-run with the feedback appended
    to ``task_prompt``. ``max_revisions`` bounds how many reject -> revise
    rounds run before giving up on the last answer. Returns a
    :class:`ProduceReviewResult` the caller can act on (e.g. escalate a
    &#8220;not approved&#8221; result to a human or a different spec).

    Give *critic_spec* a ``result_schema`` with a boolean ``approved`` field
    (and optional ``feedback``) so the loop can auto-accept; otherwise the
    critic's reply is treated as a rejection.
    """
    budget = budget or SpawnBudget()
    prompt = task_prompt
    reviews: list[DelegationResult] = []
    producer: DelegationResult | None = None
    last_result = ""

    for attempt in range(max_revisions + 1):
        producer = run_subagent(
            parent, producer_spec, prompt, budget, llm_factory=llm_factory
        )
        last_result = producer.final_message
        review_prompt = (
            f"CRITIC:\nCriteria:\n{criteria}\n\nCandidate answer:\n{last_result}"
        )
        review = run_subagent(
            parent, critic_spec, review_prompt, budget, llm_factory=llm_factory
        )
        reviews.append(review)
        approved, feedback = _critic_verdict(review)
        if approved or attempt >= max_revisions:
            return ProduceReviewResult(
                result=last_result,
                approved=approved,
                revisions=attempt,
                producer=producer,
                reviews=reviews,
            )
        prompt = (
            f"{task_prompt}\n\nA reviewer rejected your previous answer. "
            f"Revise it.\nFeedback:\n{feedback}"
        )

    # Unreachable (the loop returns), kept for type checkers.
    assert producer is not None
    return ProduceReviewResult(
        result=last_result,
        approved=False,
        revisions=max_revisions,
        producer=producer,
        reviews=reviews,
    )