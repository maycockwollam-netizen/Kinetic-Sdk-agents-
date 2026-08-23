"""Tests for FileSystemSkillLoader discovery semantics."""

from __future__ import annotations

import os

import pytest

from kinetic_sdk.skills.loader import FileSystemSkillLoader
from tests._helpers import write_skill


def test_missing_root_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        FileSystemSkillLoader(tmp_path / "does-not-exist")


def test_discover_valid_skills_sorted(tmp_path):
    write_skill(tmp_path, "beta")
    write_skill(tmp_path, "alpha")
    skills = FileSystemSkillLoader(tmp_path).discover()
    assert [s.name for s in skills] == ["alpha", "beta"]
    assert all(s.source == "local" for s in skills)


def test_broken_skill_is_skipped_not_fatal(tmp_path):
    write_skill(tmp_path, "good")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    skills = FileSystemSkillLoader(tmp_path).discover()
    assert [s.name for s in skills] == ["good"]


def test_name_mismatch_is_skipped(tmp_path):
    write_skill(tmp_path, "folder-name", frontmatter_name="other-name")
    assert FileSystemSkillLoader(tmp_path).discover() == []


def test_directory_without_skill_file_is_ignored(tmp_path):
    (tmp_path / "random-folder").mkdir()
    (tmp_path / "random-folder" / "notes.txt").write_text("x", encoding="utf-8")
    write_skill(tmp_path, "real-skill")
    skills = FileSystemSkillLoader(tmp_path).discover()
    assert [s.name for s in skills] == ["real-skill"]


def test_non_directory_entries_are_ignored(tmp_path):
    (tmp_path / "loose-file.md").write_text("x", encoding="utf-8")
    write_skill(tmp_path, "real-skill")
    skills = FileSystemSkillLoader(tmp_path).discover()
    assert [s.name for s in skills] == ["real-skill"]


def test_nested_skill_folders_are_not_recursed(tmp_path):
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    (outer / "inner" / "SKILL.md").write_text(
        "---\nname: inner\ndescription: nested\n---\nbody", encoding="utf-8"
    )
    # ``outer`` itself has no SKILL.md, and discovery is one level deep only.
    assert FileSystemSkillLoader(tmp_path).discover() == []


def test_custom_source_label(tmp_path):
    write_skill(tmp_path, "demo")
    skills = FileSystemSkillLoader(tmp_path, source_label="zip").discover()
    assert skills[0].source == "zip"


def test_symlinked_duplicate_folder_is_skipped(tmp_path, caplog):
    # The alias resolves to the same content; its frontmatter name can never
    # match the alias folder name, so it is skipped and the content is
    # discovered exactly once.
    write_skill(tmp_path, "real-skill")
    os.symlink(tmp_path / "real-skill", tmp_path / "alias-skill")
    with caplog.at_level("WARNING"):
        skills = FileSystemSkillLoader(tmp_path).discover()
    assert [s.name for s in skills] == ["real-skill"]
    assert "duplicate" in caplog.text
