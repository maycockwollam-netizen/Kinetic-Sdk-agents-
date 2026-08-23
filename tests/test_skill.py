"""Tests for the Skill dataclass: frontmatter parsing and lazy accessors.

All fixtures live in pytest's tmp_path — no real system files are touched.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.skills.exceptions import SkillParseError
from kinetic_sdk.skills.skill import MAX_DESCRIPTION_LENGTH, Skill
from kinetic_sdk.workspace.manager import PathTraversalError
from tests._helpers import write_skill

# --- from_directory / frontmatter parsing ------------------------------------


def test_from_directory_parses_frontmatter(tmp_path):
    folder = write_skill(
        tmp_path,
        "demo",
        description="Does demos.",
        body="# Demo\n\nFull body text.",
        frontmatter_extras={"version": "1.2.3"},
    )
    skill = Skill.from_directory(folder, source="local")
    assert skill.name == "demo"
    assert skill.description == "Does demos."
    assert skill.source == "local"
    assert skill.version == "1.2.3"
    assert skill.read_main() == "# Demo\n\nFull body text."


def test_version_is_optional(tmp_path):
    folder = write_skill(tmp_path, "demo")
    assert Skill.from_directory(folder, source="local").version is None


def test_missing_frontmatter_raises(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(SkillParseError, match="frontmatter"):
        Skill.from_directory(folder, source="local")


def test_unclosed_frontmatter_raises(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: demo\n", encoding="utf-8")
    with pytest.raises(SkillParseError, match="never closed"):
        Skill.from_directory(folder, source="local")


def test_missing_skill_file_raises(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    with pytest.raises(SkillParseError, match="SKILL.md"):
        Skill.from_directory(folder, source="local")


def test_missing_description_raises(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: demo\n---\nbody", encoding="utf-8")
    with pytest.raises(SkillParseError, match="description"):
        Skill.from_directory(folder, source="local")


def test_frontmatter_name_must_match_directory(tmp_path):
    folder = write_skill(tmp_path, "demo", frontmatter_name="other-name")
    with pytest.raises(SkillParseError, match="does not match"):
        Skill.from_directory(folder, source="local")


@pytest.mark.parametrize("bad", ["Demo", "-demo", "demo-", "de--mo", "de_mo", "x" * 65])
def test_invalid_names_are_rejected(tmp_path, bad):
    with pytest.raises(SkillParseError, match="invalid skill name"):
        Skill(name=bad, description="d", source="local", root_path=str(tmp_path))


def test_empty_description_is_rejected(tmp_path):
    with pytest.raises(SkillParseError, match="non-empty"):
        Skill(name="demo", description="   ", source="local", root_path=str(tmp_path))


def test_long_description_is_truncated_with_notice(tmp_path):
    description = "x" * (MAX_DESCRIPTION_LENGTH + 100)
    skill = Skill(
        name="demo", description=description, source="local", root_path=str(tmp_path)
    )
    assert len(skill.description) == MAX_DESCRIPTION_LENGTH
    assert skill.description.endswith("...[truncated, see full text at local]")


# --- read_resource / list_resources -------------------------------------------


@pytest.fixture
def rich_skill(tmp_path) -> Skill:
    folder = write_skill(
        tmp_path,
        "rich",
        body="Body.",
        resources={
            "scripts/run.sh": "echo hi",
            "references/notes.md": "notes",
            "assets/data.json": "{}",
            "internal/secret-plan.md": "hidden",
        },
    )
    return Skill.from_directory(folder, source="local")


def test_read_resource_returns_content(rich_skill):
    assert rich_skill.read_resource("scripts/run.sh") == "echo hi"


def test_read_resource_blocks_traversal(rich_skill):
    with pytest.raises(PathTraversalError):
        rich_skill.read_resource("../../etc/passwd")


def test_read_resource_outside_resource_dirs_is_rejected(rich_skill):
    # ``internal/`` exists on disk but is not one of the three resource dirs.
    with pytest.raises(PathTraversalError, match="resource"):
        rich_skill.read_resource("internal/secret-plan.md")
    with pytest.raises(PathTraversalError, match="resource"):
        rich_skill.read_resource("SKILL.md")


def test_read_resource_missing_file(rich_skill):
    with pytest.raises(FileNotFoundError, match="scripts/nope.sh"):
        rich_skill.read_resource("scripts/nope.sh")


def test_list_resources_scoped_to_three_dirs(rich_skill):
    resources = rich_skill.list_resources()
    assert resources == ["assets/data.json", "references/notes.md", "scripts/run.sh"]


def test_list_resources_pattern_filter(rich_skill):
    assert rich_skill.list_resources(pattern="*.sh") == ["scripts/run.sh"]


def test_list_resources_type_filter(rich_skill):
    assert rich_skill.list_resources(resource_type="scripts") == ["scripts/run.sh"]
    assert rich_skill.list_resources(resource_type="assets") == ["assets/data.json"]


# --- removed directory (zip-loader lifecycle contract) -------------------------


def test_read_main_after_directory_removed_raises_clearly(tmp_path):
    folder = write_skill(tmp_path, "demo", body="Body.")
    skill = Skill.from_directory(folder, source="local")
    for path in sorted(folder.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    folder.rmdir()
    with pytest.raises(SkillParseError, match="not readable"):
        skill.read_main()
