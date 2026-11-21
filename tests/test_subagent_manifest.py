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


def test_to_markdown_round_trips_through_loader(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_spec

    spec = SubagentSpec(
        name="code-reviewer",
        system_prompt="Review one diff.",
        description="Reviews a single diff.",
        model="cheap-model",
    )
    path = tmp_path / "code-reviewer.md"
    path.write_text(spec.to_markdown(), encoding="utf-8")

    loaded = load_subagent_spec(path)
    assert loaded == spec


def test_to_markdown_omits_empty_fields(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_spec

    spec = SubagentSpec(name="researcher", system_prompt="Search and report.")
    path = tmp_path / "researcher.md"
    path.write_text(spec.to_markdown(), encoding="utf-8")
    assert "model:" not in path.read_text(encoding="utf-8")
    assert load_subagent_spec(path) == spec


def test_load_subagent_specs_discovers_and_skips_broken(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_specs

    (tmp_path / "alpha.md").write_text(
        SubagentSpec(name="alpha", system_prompt="A", description="first").to_markdown(),
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text("not a spec at all\n", encoding="utf-8")

    specs = load_subagent_specs(tmp_path)
    assert list(specs) == ["alpha"]
    assert specs["alpha"].description == "first"


def test_load_subagent_specs_missing_dir_raises(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_specs

    with pytest.raises(ValueError):
        load_subagent_specs(tmp_path / "does-not-exist")


def test_tools_allowlist_round_trips_through_markdown(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_spec

    spec = SubagentSpec(
        name="reviewer",
        system_prompt="Review diffs.",
        tools=("glob", "grep", "file_editor"),
    )
    path = tmp_path / "reviewer.md"
    path.write_text(spec.to_markdown(), encoding="utf-8")

    loaded = load_subagent_spec(path)
    assert loaded == spec
    assert loaded.tools == ("glob", "grep", "file_editor")


def test_tools_allowlist_parsed_from_frontmatter(tmp_path) -> None:
    from kinetic_sdk.subagent.manifest import load_subagent_spec

    path = tmp_path / "reader.md"
    path.write_text(
        "---\nname: reader\ndescription: read only\ntools: glob, grep\n---\n\nRead files.\n",
        encoding="utf-8",
    )
    loaded = load_subagent_spec(path)
    assert loaded.tools == ("glob", "grep")
