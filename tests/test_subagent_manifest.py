"""Tests for subagent/manifest.py — SubagentSpec validation."""

from __future__ import annotations

import dataclasses

import pytest

from kinetic_sdk.subagent import (
    SUBAGENT_MAX_NAME_LENGTH,
    SubagentSpec,
    SubagentSpecError,
)


def test_valid_minimal_spec() -> None:
    spec = SubagentSpec(name="researcher", system_prompt="You research one question.")
    assert spec.name == "researcher"
    assert spec.system_prompt == "You research one question."
    assert spec.description == ""
    assert spec.model is None


def test_valid_full_spec() -> None:
    spec = SubagentSpec(
        name="code-reviewer-2",
        system_prompt="You review diffs.",
        description="Reviews one diff.",
        model="cheap-model",
    )
    assert spec.description == "Reviews one diff."
    assert spec.model == "cheap-model"


def test_spec_is_frozen() -> None:
    spec = SubagentSpec(name="researcher", system_prompt="prompt")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "Researcher",  # uppercase
        "-researcher",  # leading hyphen
        "researcher-",  # trailing hyphen
        "re--searcher",  # double hyphen
        "re searcher",  # space
        "re_searcher",  # underscore
        "a" * (SUBAGENT_MAX_NAME_LENGTH + 1),  # too long
    ],
)
def test_invalid_names_rejected(bad_name: str) -> None:
    with pytest.raises(SubagentSpecError, match="Invalid sub-agent name"):
        SubagentSpec(name=bad_name, system_prompt="prompt")


def test_name_at_max_length_accepted() -> None:
    name = "a" * SUBAGENT_MAX_NAME_LENGTH
    spec = SubagentSpec(name=name, system_prompt="prompt")
    assert spec.name == name


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\n\t "])
def test_blank_system_prompt_rejected(bad_prompt: str) -> None:
    with pytest.raises(SubagentSpecError, match="non-empty system_prompt"):
        SubagentSpec(name="researcher", system_prompt=bad_prompt)


def test_spec_error_is_a_value_error() -> None:
    # Callers that only catch ValueError (like skill/plugin manifest
    # validation) must also catch spec errors.
    with pytest.raises(ValueError):
        SubagentSpec(name="BAD", system_prompt="prompt")
