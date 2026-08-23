"""Tests for SkillRegistry: the vetting invariant and precedence rules."""

from __future__ import annotations

import pytest

from kinetic_sdk.skills.exceptions import SkillNotFoundError, SkillVetError
from kinetic_sdk.skills.loader import FileSystemSkillLoader
from kinetic_sdk.skills.registry import SkillRegistry
from kinetic_sdk.skills.skill import Skill
from kinetic_sdk.skills.vet import VetFlag, VetResult
from tests._helpers import write_skill


def _local_skill(tmp_path, name: str, description: str = "A demo skill.") -> Skill:
    return Skill.from_directory(write_skill(tmp_path, name, description=description), source="local")


def _external_skill(tmp_path, name: str, source: str = "zip") -> Skill:
    folder = write_skill(tmp_path, name)
    return Skill.from_directory(folder, source=source)


CLEAN = VetResult(clean=True, flags=[], reviewed_by_llm=False)
DIRTY = VetResult(
    clean=False,
    flags=[
        VetFlag(
            severity="critical",
            category="prompt_injection",
            message="bad",
            location="SKILL.md",
        )
    ],
    reviewed_by_llm=False,
)


# --- the vetting invariant ------------------------------------------------------


def test_local_skill_needs_no_vet_result(tmp_path):
    registry = SkillRegistry()
    registry.add(_local_skill(tmp_path, "demo"))
    assert [s.name for s in registry.list_skills()] == ["demo"]


def test_non_local_without_vet_result_is_rejected(tmp_path):
    registry = SkillRegistry()
    with pytest.raises(SkillVetError, match="without a VetResult"):
        registry.add(_external_skill(tmp_path, "zip-skill"))
    assert registry.list_skills() == []


def test_non_local_with_failed_vet_is_rejected(tmp_path):
    registry = SkillRegistry()
    with pytest.raises(SkillVetError, match="failed vetting"):
        registry.add(_external_skill(tmp_path, "zip-skill"), vet_result=DIRTY)
    assert registry.list_skills() == []


def test_non_local_with_clean_vet_is_accepted(tmp_path):
    registry = SkillRegistry()
    registry.add(_external_skill(tmp_path, "zip-skill"), vet_result=CLEAN)
    assert [s.name for s in registry.list_skills()] == ["zip-skill"]


def test_add_many_shares_one_vet_result(tmp_path):
    registry = SkillRegistry()
    skills = [_external_skill(tmp_path, "one"), _external_skill(tmp_path, "two")]
    registry.add_many(skills, vet_result=CLEAN)
    assert sorted(s.name for s in registry.list_skills()) == ["one", "two"]


# --- precedence: first registration wins ---------------------------------------


def test_first_registration_wins(tmp_path, caplog):
    registry = SkillRegistry()
    first = _local_skill(tmp_path, "demo", description="first")
    second = _local_skill(tmp_path / "elsewhere", "demo", description="second")
    with caplog.at_level("WARNING"):
        registry.add(first)
        registry.add(second)
    assert registry.get("demo").description == "first"
    assert "already registered" in caplog.text


# --- load_from ---------------------------------------------------------------------


def test_load_from_local_loader(tmp_path):
    write_skill(tmp_path, "alpha")
    write_skill(tmp_path, "beta")
    registry = SkillRegistry()
    discovered = registry.load_from(FileSystemSkillLoader(tmp_path))
    assert sorted(s.name for s in discovered) == ["alpha", "beta"]
    assert sorted(s.name for s in registry.list_skills()) == ["alpha", "beta"]


# --- access -------------------------------------------------------------------------


def test_get_unknown_name_raises(tmp_path):
    registry = SkillRegistry()
    with pytest.raises(SkillNotFoundError, match="'nope'"):
        registry.get("nope")


def test_load_unknown_name_raises(tmp_path):
    registry = SkillRegistry()
    with pytest.raises(SkillNotFoundError, match="'nope'"):
        registry.load("nope")


def test_load_returns_full_body(tmp_path):
    registry = SkillRegistry()
    registry.add(_local_skill(tmp_path, "demo"))
    assert registry.load("demo") == "Demo body."


def test_to_prompt_catalog_is_sorted_and_compact(tmp_path):
    registry = SkillRegistry()
    registry.add(_local_skill(tmp_path, "beta", description="B skill."))
    registry.add(_local_skill(tmp_path, "alpha", description="A skill."))
    assert registry.to_prompt_catalog() == "- alpha: A skill.\n- beta: B skill."
