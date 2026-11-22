"""Ready-made LLM-backed answer verifier for the agent's verification loop.

``AnswerVerifier`` (``agent/planning.py``) is the protocol SDK users implement
to check a proposed final answer; the agent's verification loop turns a
rejection into an in-context correction round (bounded by
``max_verification_retries``) and a stuck rejection into
:class:`AnswerNotVerifiedError`. Until now no implementation shipped, so the
loop was unusable out of the box.

:class:`LLMAnswerVerifier` fills that gap: it asks a SEPARATE judge model
(never the agent's actor model) whether the answer satisfies the task, and
returns an accepted/rejected verdict whose ``feedback`` feeds the existing
correction loop. The task/answer/plan are redacted before they leave the SDK.
"""

from __future__ import annotations

from kinetic_sdk.agent.planning import Plan, VerificationResult
from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.security.redact import redact_value

_VERIFY_PROMPT = (
    "You are a strict answer verifier. Decide whether the assistant's answer "
    "satisfies the user's task, using the execution plan as context. "
    "For software tasks, do NOT accept an answer that merely claims success: "
    "require evidence (tests run, files changed) where the task needs it.\n\n"
    "Reply with EXACTLY one line starting with either:\n"
    "- ACCEPTED: <one short note>\n- REJECTED: <specific, actionable reason>\n\n"
    "User task:\n{task}\n\n"
    "Execution plan:\n{plan}\n\n"
    "Assistant answer:\n{answer}"
)


class LLMAnswerVerifier:
    """Verify a final answer with a distinct judge model.

    Args:
        llm: Any :class:`~kinetic_sdk.llm.client.LLMClient` — meant to be a
            model separate from the agent's actor model (same rule as the
            classifier and the memory consolidator).
        max_tokens: Reply budget for the one-line verdict.

    Behaviour contract (matches how the agent treats verifiers):

    * An ``ACCEPTED``-prefixed reply → accepted, empty feedback.
    * An ``REJECTED``-prefixed reply → rejected with the remainder of that
      line as actionable feedback (fed back by the correction loop).
    * Any other reply → treated as a rejection ("unparseable verdict"), so a
      confused verifier can never silently approve an answer.
    * A raised :class:`Exception` (network/provider) propagates to the agent,
      which applies ``verifier_fail_closed`` (default: reject).
    """

    _VERIFY_PROMPT = _VERIFY_PROMPT

    def __init__(self, llm: LLMClient, *, max_tokens: int = 300) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._llm = llm
        self._max_tokens = max_tokens

    def verify(self, task: str, answer: str, plan: Plan | None) -> VerificationResult:
        """Judge *answer* against *task*; see class docstring for the contract."""
        prompt = self._VERIFY_PROMPT.format(
            task=redact_value(task),
            plan=redact_value(plan.to_prompt()) if plan is not None else "(no plan)",
            answer=redact_value(answer),
        )
        response = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return self._parse(response.content or "")

    @staticmethod
    def _parse(text: str) -> VerificationResult:
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        upper = first_line.upper()
        if upper.startswith("REJECTED"):
            feedback = (
                first_line.split(":", 1)[1].strip()
                if ":" in first_line
                else "Verifier rejected the answer."
            )
            return VerificationResult(accepted=False, feedback=feedback)
        if upper.startswith("ACCEPTED"):
            return VerificationResult(accepted=True, feedback="")
        return VerificationResult(
            accepted=False,
            feedback="Verifier returned an unparseable verdict; "
            "treat the answer as unverified.",
        )