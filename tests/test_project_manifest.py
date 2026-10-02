from __future__ import annotations

import pytest

from kinetic_sdk.project import ManifestError, ProjectManifest, load_project_manifest


def test_loads_strict_project_manifest(tmp_path):
    (tmp_path / ".kinetic").mkdir()
    (tmp_path / ".kinetic/project.toml").write_text(
        '''setup = "python -m pip install -e ."
test = "python -m pytest -q"
test_targeted = "python -m pytest -q {files}"
lint = "ruff check kinetic_sdk"
typecheck = "mypy"
protected_paths = [".github/", "pyproject.toml"]
command_timeout = 90
max_diff_lines = 400
[verification]
kind = "tests"
''',
        encoding="utf-8",
    )

    result = load_project_manifest(tmp_path)

    assert isinstance(result, ProjectManifest)
    assert result.test_targeted == "python -m pytest -q {files}"
    assert result.verification.kind == "tests"


def test_missing_manifest_returns_clear_status(tmp_path):
    result = load_project_manifest(tmp_path)
    assert result.status == "no_manifest"
    assert result.manifest is None


def test_manifest_rejects_unknown_and_invalid_keys(tmp_path):
    path = tmp_path / "project.toml"
    path.write_text("test = 'pytest'\nunknown = true\n[verification]\nkind = 'tests'\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="unknown"):
        ProjectManifest.from_file(path)
