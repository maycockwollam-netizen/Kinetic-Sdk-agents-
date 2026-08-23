"""Tests for kinetic_sdk.plugin.discovery."""
from __future__ import annotations

import sys

import pytest

from kinetic_sdk.plugin import PluginManifestError
from kinetic_sdk.plugin.discovery import (
    discover_directory_plugins,
    discover_entry_point_plugins,
    discover_plugins,
)

from ._helpers import FakeEntryPoint, write_plugin


class TestEntryPointDiscovery:
    def test_discovers_entry_points(self):
        eps = [
            FakeEntryPoint("alpha", "pkg_alpha.tools:build", "1.2"),
            FakeEntryPoint("beta", "pkg_beta:make_tools"),
        ]
        manifests = discover_entry_point_plugins(lambda: eps)
        assert [m.name for m in manifests] == ["alpha", "beta"]
        assert manifests[0].version == "1.2"
        assert manifests[0].entry_point == "pkg_alpha.tools:build"
        assert all(m.source == "entry_point" for m in manifests)

    def test_entry_point_without_dist_version(self):
        manifests = discover_entry_point_plugins(lambda: [FakeEntryPoint("a", "m:f")])
        assert manifests[0].version is None

    def test_empty_group_yields_nothing(self):
        assert discover_entry_point_plugins(lambda: []) == []

    def test_does_not_import_anything(self):
        # A provider returning entry points for modules that do not exist
        # must succeed — discovery resolves no imports.
        eps = [FakeEntryPoint("ghost", "no_such_module_anywhere:build")]
        manifests = discover_entry_point_plugins(lambda: eps)
        assert manifests[0].module_spec == "no_such_module_anywhere"
        assert "no_such_module_anywhere" not in sys.modules


class TestDirectoryDiscovery:
    def test_discovers_plugin_dirs(self, tmp_path):
        write_plugin(tmp_path, "alpha")
        write_plugin(tmp_path, "beta")
        manifests = discover_directory_plugins(tmp_path)
        assert [m.name for m in manifests] == ["alpha", "beta"]
        assert all(m.source == "directory" for m in manifests)

    def test_dirs_without_plugin_md_skipped(self, tmp_path):
        (tmp_path / "not-a-plugin").mkdir()
        write_plugin(tmp_path, "real")
        manifests = discover_directory_plugins(tmp_path)
        assert [m.name for m in manifests] == ["real"]

    def test_malformed_plugin_warns_and_skips(self, tmp_path, caplog):
        write_plugin(tmp_path, "good")
        write_plugin(tmp_path, "bad", frontmatter_name="mismatch")
        with caplog.at_level("WARNING"):
            manifests = discover_directory_plugins(tmp_path)
        assert [m.name for m in manifests] == ["good"]
        assert any("bad" in record.message for record in caplog.records)

    def test_missing_directory_warns_and_returns_empty(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            manifests = discover_directory_plugins(tmp_path / "nope")
        assert manifests == []
        assert caplog.records

    def test_discovery_does_not_import_plugin_code(self, tmp_path):
        marker = tmp_path / "imported.marker"
        module = (
            f"from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('side effect ran')\n"
            "def make_tools():\n    return []\n"
        )
        write_plugin(tmp_path, "sneaky", modules={"plugin.py": module})
        discover_directory_plugins(tmp_path)
        assert not marker.exists()
        assert not any(
            name.startswith("kinetic_plugin_") for name in sys.modules
        )


class TestMergedDiscovery:
    def test_merges_both_sources(self, tmp_path):
        write_plugin(tmp_path, "from-dir")
        eps = [FakeEntryPoint("from-ep", "pkg:build")]
        manifests = discover_plugins(directory=tmp_path, entry_points_provider=lambda: eps)
        by_name = {m.name: m for m in manifests}
        assert set(by_name) == {"from-dir", "from-ep"}
        assert by_name["from-dir"].source == "directory"
        assert by_name["from-ep"].source == "entry_point"

    def test_duplicate_name_across_sources_raises(self, tmp_path):
        write_plugin(tmp_path, "clash")
        eps = [FakeEntryPoint("clash", "pkg:build")]
        with pytest.raises(PluginManifestError, match="clash"):
            discover_plugins(directory=tmp_path, entry_points_provider=lambda: eps)

    def test_no_directory_source_when_none(self):
        eps = [FakeEntryPoint("only-ep", "pkg:build")]
        manifests = discover_plugins(directory=None, entry_points_provider=lambda: eps)
        assert [m.name for m in manifests] == ["only-ep"]
