"""Tests for auto-generated ``repo`` skills (codemap/skill.py)."""

from __future__ import annotations

import pytest

from kinetic_sdk.codemap.skill import build_repo_skill
from kinetic_sdk.skills.skill import Skill


def _sample_repo(tmp_path) -> object:
    root = tmp_path / "sample-project"
    root.mkdir()
    (root / "README.md").write_text("# Sample\n\nA tiny project for tests.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("requires=[]\n[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("## Conventions\nUse type hints everywhere.\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / ".hidden").write_text("secret", encoding="utf-8")
    return root


def test_build_repo_skill_writes_valid_skill(tmp_path) -> None:
    root = _sample_repo(tmp_path)
    out = tmp_path / "out"

    skill = build_repo_skill(root, out_root=out)

    assert skill.name == "sample-project"
    assert skill.type == "repo"
    assert skill.source == "generated"
    assert skill.description == "A tiny project for tests."

    skill_md = out / "sample-project" / "SKILL.md"
    assert skill_md.is_file()
    body = skill_md.read_text(encoding="utf-8")
    assert "## Repository layout" in body
    assert "`src/`" in body
    assert "`tests/`" in body
    assert "pytest" in body
    assert "Use type hints" in body  # AGENTS.md conventions included
    assert ".hidden" not in body     # hidden files excluded

    # The written skill reloads cleanly through the normal loader.
    reloaded = Skill.from_directory(out / "sample-project", source="generated")
    assert reloaded == skill


def test_build_repo_skill_reads_skill_md_body(tmp_path) -> None:
    root = _sample_repo(tmp_path)
    skill = build_repo_skill(root, out_root=tmp_path / "o")
    assert skill.read_main()  # body is non-empty


def test_build_repo_skill_respects_explicit_name_and_description(tmp_path) -> None:
    root = _sample_repo(tmp_path)
    skill = build_repo_skill(
        root, name="my-aliases", description="Custom description.", out_root=tmp_path / "o"
    )
    assert skill.name == "my-aliases"
    assert skill.description == "Custom description."
    assert (tmp_path / "o" / "my-aliases" / "SKILL.md").is_file()


def test_build_repo_skill_rejects_bad_inputs(tmp_path) -> None:

    root = _sample_repo(tmp_path)
    with pytest.raises(ValueError):
        build_repo_skill(tmp_path / "missing")
    with pytest.raises(ValueError):
        build_repo_skill(root, name="Not Valid Name", out_root=tmp_path / "o")


def test_build_repo_skill_on_repo_without_docs(tmp_path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    skill = build_repo_skill(bare, out_root=tmp_path / "o")
    assert skill.description  # non-empty fallback description
    body = skill.read_main()
    assert "No AGENTS.md" in body