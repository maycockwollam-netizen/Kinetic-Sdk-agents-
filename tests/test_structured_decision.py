"""Tests for :mod:`kinetic_sdk.agent.decision` (typed structured decisions)."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.decision import StructuredDecision
from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.testing import MockLLMClient, text_response


def _decider(*, llm: LLMClient, choices=("yes", "no"), fallback="yes", **kwargs) -> StructuredDecision:
    return StructuredDecision(
        llm=llm,
        question="Does this need a file write?",
        choices=choices,
        fallback=fallback,
        **kwargs,
    )


def test_exact_match_returns_choice_with_full_confidence() -> None:
    decider = _decider(llm=MockLLMClient([text_response("no")]))
    outcome = decider.decide("create a file")
    assert outcome.value == "no"
    assert outcome.confidence == 1.0
    assert outcome.rationale == "direct"


def test_answer_is_normalised_before_matching() -> None:
    decider = _decider(llm=MockLLMClient([text_response("Yes.")]))
    assert decider.decide("x").value == "yes"


def test_choice_found_inside_larger_answer() -> None:
    decider = _decider(llm=MockLLMClient([text_response("no, not needed")]))
    assert decider.decide("x").value == "no"


def test_retries_with_correction_then_accepts() -> None:
    decider = _decider(
        llm=MockLLMClient([text_response("maybe"), text_response("no")]),
        max_retries=1,
    )
    outcome = decider.decide("x")
    assert outcome.value == "no"
    assert outcome.confidence == 0.5
    assert outcome.rationale == "retry"


def test_falls_back_after_retries_exhausted() -> None:
    decider = _decider(llm=MockLLMClient([text_response("maybe")]), max_retries=1)
    outcome = decider.decide("x")
    assert outcome.value == "yes"  # fallback
    assert outcome.confidence == 0.0
    assert outcome.rationale == "unparseable"


def test_model_error_falls_back_to_safe_value() -> None:
    class Broken(LLMClient):
        def chat(self, messages, tools=None, system=None, **kwargs):
            raise RuntimeError("offline")

    outcome = _decider(llm=Broken()).decide("x")
    assert outcome.value == "yes"
    assert outcome.confidence == 0.0
    assert outcome.rationale == "unparseable"


@pytest.mark.parametrize(
    ("choices", "fallback"),
    [([], "x"), (["a", "b"], "c")],
)
def test_invalid_configuration_rejected(choices, fallback) -> None:
    with pytest.raises(ValueError):
        StructuredDecision(
            llm=MockLLMClient([]),
            question="q",
            choices=choices,
            fallback=fallback,
        )


def test_prompt_includes_intro_question_and_choices() -> None:
    from kinetic_sdk.testing import MockLLMClient

    llm = MockLLMClient([text_response("yes")])
    decider = _decider(llm=llm, intro="Classify it.")
    decider.decide("some context")
    message = llm.calls[0]["messages"][-1]["content"]
    assert "Classify it." in message
    assert "Does this need a file write?" in message
    assert "yes" in message and "no" in message