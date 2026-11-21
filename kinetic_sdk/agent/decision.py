"""Model-backed structured decisions (typed closed questions).

Inspired by opencode's ``DecisionClient``: instead of asking the model for
free-form text, pose a *closed* question with a fixed list of choices and
coerce the reply to exactly one of them, with a safe fallback. This makes
lightweight routing / classification cheap, reliable and easy to extend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from kinetic_sdk.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    """One coerced choice from a :class:`StructuredDecision` query.

    Attributes:
        value: One of the declared ``choices``, or ``fallback`` when the
            model could not answer.
        confidence: In ``[0, 1]`` - ``1.0`` for an exact match, ``0.5`` for a
            match after a self-correction retry, ``0.0`` for a fallback.
        rationale: Short note for debugging/logging (never secrets).
    """

    value: str
    confidence: float
    rationale: str


def _normalise(text: str) -> str:
    return text.strip().strip(".,;:'\"`)]}").lower()


class StructuredDecision:
    """Ask an LLM a closed question and coerce the reply to one choice.

    Args:
        llm: Any :class:`~kinetic_sdk.llm.client.LLMClient` (a cheap model is
            the usual choice).
        question: The closed question to answer.
        choices: The allowed answers, in order. Shown verbatim to the model.
        fallback: Value returned when the model cannot answer; must be one of
            ``choices``.
        intro: Optional framing sentence (e.g. "Classify the user request.")
            prepended to the prompt.
        max_tokens: Small token budget for the answer.
        max_retries: How many self-correction retries after an unparseable
            answer (default 1).

    Example::

        decider = StructuredDecision(
            llm=cheap_client,
            question="Does this task need to write files?",
            choices=["yes", "no"],
            fallback="yes",
        )
        outcome = decider.decide(task_text)
        if outcome.value == "yes": ...
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        question: str,
        choices: Sequence[str],
        fallback: str,
        intro: str | None = None,
        max_tokens: int = 32,
        max_retries: int = 1,
    ) -> None:
        if not choices:
            raise ValueError("choices must be non-empty")
        if fallback not in choices:
            raise ValueError("fallback must be one of the declared choices")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._llm = llm
        self._question = question
        self._choices = list(dict.fromkeys(choices))
        self._fallback = fallback
        self._intro = intro
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._accept = {_normalise(choice): choice for choice in self._choices}

    @property
    def choices(self) -> list[str]:
        """The allowed choices, in declaration order."""
        return list(self._choices)

    def decide(self, context: str) -> Decision:
        """Ask the model and return a coerced :class:`Decision`.

        Never raises for model failures: any error or unparseable reply falls
        back to :attr:`fallback`, mirroring the classifier's safe-side default.
        """
        prompt = self._build_prompt(context)
        for attempt in range(self._max_retries + 1):
            matched = self._call_and_match(prompt)
            if matched is not None:
                return Decision(
                    matched,
                    1.0 if attempt == 0 else 0.5,
                    "direct" if attempt == 0 else "retry",
                )
            if attempt < self._max_retries:
                prompt = prompt + [
                    {
                        "role": "user",
                        "content": (
                            "Invalid answer. Reply with exactly ONE of: "
                            + ", ".join(self._choices)
                            + "."
                        ),
                    }
                ]
        return Decision(self._fallback, 0.0, "unparseable")

    def _call_and_match(self, prompt: list[dict[str, Any]]) -> str | None:
        try:
            response = self._llm.chat(messages=prompt, max_tokens=self._max_tokens)
            return self._match(_normalise(response.content or ""))
        except Exception as exc:  # noqa: BLE001 - safe-side fallback
            logger.warning("StructuredDecision call failed: %s", type(exc).__name__)
            return None

    def _match(self, answer: str) -> str | None:
        if answer in self._accept:
            return self._accept[answer]
        for token in answer.replace(",", " ").split():
            if token in self._accept:
                return self._accept[token]
        return None

    def _build_prompt(self, context: str) -> list[dict[str, Any]]:
        lines = ([] if self._intro is None else [self._intro + "\n"]) + [
            self._question,
            "Reply with exactly ONE of the following: " + ", ".join(self._choices) + ".",
        ]
        if context:
            lines.append("\nContext:\n" + context)
        return [{"role": "user", "content": "\n".join(lines)}]