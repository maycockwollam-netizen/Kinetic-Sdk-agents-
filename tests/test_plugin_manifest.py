"""Tests for kinetic_sdk.plugin.manifest (PluginManifest validation)."""
from __future__ import annotations

import pytest

from kinetic_sdk.plugin import PluginManifest, PluginManifestError
from kinetic_sdk.skills.skill import MAX_NAME_LENGTH

from ._helpers import write_plugin


def make_manifest(**overrides) -> PluginManifest:
    kwargs = {
        "name": "demo",
        "version": "1.0",
        "entry_point": "plugin:make_tools",
        "source": "entry_point",
        "declared_capabilities": frozenset({"tool"}),
    }
    kwargs.update(overrides)
    return PluginManifest(**kwargs)


class TestNameValidation:
    def test_valid_name(self):
        assert make_manifest(name="my-plugin-2").name == "my-plugin-2"

    @pytest.mark.parametrize(
        "bad", ["Demo", "-demo", "demo-", "a--b", "my_plugin", "", "a" * (MAX_NAME_LENGTH + 1)]
    )
    def test_invalid_names_raise(self, bad):
        with pytest.raises(PluginManifestError):
            make_manifest(name=bad)


class TestEntryPointValidation:
    @pytest.mark.parametrize(
        "good", ["plugin:make_tools", "pkg.sub.mod:factory", "_x:_y"]
    )
    def test_valid_entry_points(self, good):
        assert make_manifest(entry_point=good).entry_point == good

    @pytest.mark.parametrize(
        "bad",
        ["plugin", "plugin:", ":make_tools", "plugin:make tools", "9lives:f", "p-k:f"],
    )
    def test_invalid_entry_points_raise(self, bad):
        with pytest.raises(PluginManifestError):
            make_manifest(entry_point=bad)

    def test_module_spec_and_factory_name(self):
        manifest = make_manifest(entry_point="pkg.tools:build")
        assert manifest.module_spec == "pkg.tools"
        assert manifest.factory_name == "build"


class TestCapabilitiesValidation:
    def test_tool_and_hook_allowed(self):
        manifest = make_manifest(declared_capabilities=frozenset({"tool", "hook"}))
        assert manifest.declared_capabilities == frozenset({"tool", "hook"})

    def test_skill_capability_is_allowed(self):
        manifest = make_manifest(declared_capabilities=frozenset({"tool", "skill"}))
        assert manifest.declared_capabilities == frozenset({"tool", "skill"})

    def test_empty_capabilities_raise(self):
        with pytest.raises(PluginManifestError):
            make_manifest(declared_capabilities=frozenset())

    def test_unknown_capability_raises(self):
        with pytest.raises(PluginManifestError):
            make_manifest(declared_capabilities=frozenset({"tool", "invalid"}))


class TestSourceAndDirectory:
    def test_directory_source_requires_directory(self):
        with pytest.raises(PluginManifestError):
            make_manifest(source="directory", directory=None)

    def test_entry_point_source_forbids_directory(self, tmp_path):
        with pytest.raises(PluginManifestError):
            make_manifest(source="entry_point", directory=str(tmp_path))

    def test_invalid_source_raises(self):
        with pytest.raises(PluginManifestError):
            make_manifest(source="local")

    def test_version_must_be_nonempty(self):
        with pytest.raises(PluginManifestError):
            make_manifest(version="   ")

    def test_version_may_be_none(self):
        assert make_manifest(version=None).version is None


class TestFromDirectory:
    def test_parses_frontmatter(self, tmp_path):
        directory = write_plugin(tmp_path, "demo")
        manifest = PluginManifest.from_directory(directory)
        assert manifest.name == "demo"
        assert manifest.version == "0.1"
        assert manifest.entry_point == "plugin:make_tools"
        assert manifest.source == "directory"
        assert manifest.declared_capabilities == frozenset({"tool"})
        assert manifest.directory == str(directory)

    def test_missing_plugin_md_raises(self, tmp_path):
        (tmp_path / "demo").mkdir()
        with pytest.raises(PluginManifestError, match="PLUGIN.md"):
            PluginManifest.from_directory(tmp_path / "demo")

    def test_name_must_match_directory(self, tmp_path):
        directory = write_plugin(tmp_path, "demo", frontmatter_name="other")
        with pytest.raises(PluginManifestError, match="does not match"):
            PluginManifest.from_directory(directory)

    def test_missing_entry_point_raises(self, tmp_path):
        directory = tmp_path / "demo"
        directory.mkdir()
        (directory / "PLUGIN.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
        with pytest.raises(PluginManifestError, match="entry_point"):
            PluginManifest.from_directory(directory)

    def test_malformed_frontmatter_raises(self, tmp_path):
        directory = tmp_path / "demo"
        directory.mkdir()
        (directory / "PLUGIN.md").write_text("no frontmatter here\n", encoding="utf-8")
        with pytest.raises(PluginManifestError):
            PluginManifest.from_directory(directory)

    def test_capabilities_parsed_from_frontmatter(self, tmp_path):
        directory = write_plugin(tmp_path, "demo", capabilities="tool, hook")
        manifest = PluginManifest.from_directory(directory)
        assert manifest.declared_capabilities == frozenset({"tool", "hook"})

    def test_docker_pids_limit_is_parsed_from_frontmatter(self, tmp_path):
        directory = write_plugin(tmp_path, "demo")
        plugin_file = directory / "PLUGIN.md"
        plugin_file.write_text(
            plugin_file.read_text(encoding="utf-8").replace(
                "entry_point: plugin:make_tools", "entry_point: plugin:make_tools\npids_limit: 42"
            ),
            encoding="utf-8",
        )
        assert PluginManifest.from_directory(directory).pids_limit == 42


@pytest.mark.parametrize("pids_limit", [0, -1])
def test_pids_limit_must_be_positive(pids_limit: int):
    with pytest.raises(PluginManifestError, match="resource limits must be positive"):
        make_manifest(pids_limit=pids_limit)


class TestFromEntryPoint:
    def test_builds_manifest(self):
        manifest = PluginManifest.from_entry_point("demo", "pkg.tools:build", "2.0")
        assert manifest.name == "demo"
        assert manifest.version == "2.0"
        assert manifest.source == "entry_point"
        assert manifest.declared_capabilities == frozenset({"tool"})
        assert manifest.directory is None

    def test_invalid_ep_name_rejected(self):
        with pytest.raises(PluginManifestError):
            PluginManifest.from_entry_point("Bad_Name", "pkg:f")
