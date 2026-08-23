"""Tests for ZipSkillLoader: safe extraction and the lifecycle contract."""

from __future__ import annotations

import os
import stat
import tempfile
import zipfile

import pytest
from tests._helpers import make_zip, skill_zip_entries

from kinetic_sdk.skills.exceptions import SkillParseError, ZipSkillError
from kinetic_sdk.skills.zip_loader import ZipSkillLoader


def _temp_skill_dirs() -> set[str]:
    return {
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("kinetic-skills-")
    }


# --- happy path ---------------------------------------------------------------


def test_discover_skills_from_valid_zip(tmp_path):
    entries = skill_zip_entries("alpha", body="Alpha body.")
    entries.update(skill_zip_entries("beta"))
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with ZipSkillLoader(zip_path) as loader:
        skills = loader.discover()
        assert [s.name for s in skills] == ["alpha", "beta"]
        assert all(s.source == "zip" for s in skills)
        # Content is readable while the loader is open.
        assert skills[0].read_main() == "Alpha body."


def test_discover_is_idempotent(tmp_path):
    zip_path = make_zip(tmp_path / "skills.zip", skill_zip_entries("alpha"))
    with ZipSkillLoader(zip_path) as loader:
        first = loader.discover()
        second = loader.discover()
    assert [s.name for s in first] == [s.name for s in second]


def test_missing_zip_path_raises(tmp_path):
    with pytest.raises(ValueError, match="not an existing file"):
        ZipSkillLoader(tmp_path / "nope.zip")


def test_invalid_zip_raises(tmp_path):
    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"this is not a zip")
    with ZipSkillLoader(fake) as loader:
        with pytest.raises(ZipSkillError, match="not a valid zip"):
            loader.discover()


# --- lifecycle contract ---------------------------------------------------------


def test_temp_dir_removed_after_close_and_reads_fail_clearly(tmp_path):
    zip_path = make_zip(tmp_path / "skills.zip", skill_zip_entries("alpha"))
    loader = ZipSkillLoader(zip_path)
    skills = loader.discover()
    temp_root = loader._temp_dir.name
    assert os.path.isdir(temp_root)
    loader.close()
    assert not os.path.exists(temp_root)
    with pytest.raises(SkillParseError, match="not readable"):
        skills[0].read_main()
    with pytest.raises(RuntimeError, match="closed"):
        loader.discover()


def test_close_is_idempotent(tmp_path):
    zip_path = make_zip(tmp_path / "skills.zip", skill_zip_entries("alpha"))
    loader = ZipSkillLoader(zip_path)
    loader.close()
    loader.close()


def test_no_temp_dir_leaks_after_context_manager(tmp_path):
    before = _temp_skill_dirs()
    zip_path = make_zip(tmp_path / "skills.zip", skill_zip_entries("alpha"))
    with ZipSkillLoader(zip_path) as loader:
        loader.discover()
        during = _temp_skill_dirs()
    assert _temp_skill_dirs() == before
    assert len(during - before) == 1


def test_failed_extraction_cleans_up(tmp_path):
    entries = skill_zip_entries("alpha")
    entries["evil/../../escape.md"] = "x"
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    before = _temp_skill_dirs()
    with pytest.raises(ZipSkillError):
        ZipSkillLoader(zip_path).discover()
    assert _temp_skill_dirs() == before


# --- attack cases ---------------------------------------------------------------


@pytest.mark.parametrize(
    "evil_entry",
    ["evil/../../../etc/passwd.md", "/etc/passwd.md", "a/../../escape.md"],
)
def test_zip_slip_entries_are_rejected(tmp_path, evil_entry):
    entries = skill_zip_entries("alpha")
    entries[evil_entry] = "pwned"
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with ZipSkillLoader(zip_path) as loader:
        with pytest.raises(ZipSkillError, match="zip-slip"):
            loader.discover()


def test_symlink_entry_is_skipped_without_a_trace(tmp_path, caplog):
    zip_path = tmp_path / "skills.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, content in skill_zip_entries(
            "alpha", resources={"scripts/run.sh": "echo ok"}
        ).items():
            archive.writestr(name, content)
        link = zipfile.ZipInfo("alpha/scripts/evil-link.sh")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "/etc/passwd")
    with caplog.at_level("WARNING"):
        with ZipSkillLoader(zip_path) as loader:
            skills = loader.discover()
            assert [s.name for s in skills] == ["alpha"]
            # The valid sibling script survived; the symlink left nothing.
            assert skills[0].list_resources() == ["scripts/run.sh"]
            root = skills[0].root_path
            assert not os.path.exists(os.path.join(root, "scripts", "evil-link.sh"))
    assert "symlink" in caplog.text


def test_zip_bomb_declared_size_raises_before_extracting(tmp_path):
    entries = skill_zip_entries("alpha")
    entries["alpha/assets/big.md"] = "x" * 4096
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with ZipSkillLoader(zip_path, max_total_size_bytes=1024) as loader:
        with pytest.raises(ZipSkillError, match="zip-bomb"):
            loader.discover()


def test_too_many_entries_raises(tmp_path):
    entries = {f"filler/{i}.md": "" for i in range(600)}
    entries.update(skill_zip_entries("alpha"))
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with ZipSkillLoader(zip_path) as loader:
        with pytest.raises(ZipSkillError, match="entries exceeds"):
            loader.discover()


def test_disallowed_extension_is_skipped_not_fatal(tmp_path, caplog):
    entries = skill_zip_entries(
        "alpha", resources={"scripts/run.sh": "echo ok", "scripts/evil.exe": "MZ"}
    )
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with caplog.at_level("WARNING"):
        with ZipSkillLoader(zip_path) as loader:
            skills = loader.discover()
            assert [s.name for s in skills] == ["alpha"]
            assert skills[0].list_resources() == ["scripts/run.sh"]
    assert "allowlist" in caplog.text


def test_directory_entries_are_not_filtered_by_extension(tmp_path):
    # ``scripts/`` has no extension; it must survive the file-extension
    # allowlist or the whole resource layout would break.
    entries = skill_zip_entries("alpha")
    entries["alpha/scripts/"] = None
    entries["alpha/scripts/run.sh"] = "echo ok"
    zip_path = make_zip(tmp_path / "skills.zip", entries)
    with ZipSkillLoader(zip_path) as loader:
        skills = loader.discover()
        assert skills[0].list_resources() == ["scripts/run.sh"]
