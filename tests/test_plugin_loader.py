"""Tests for kinetic_sdk.plugin.loader (scan -> import -> audit)."""
from __future__ import annotations

import sys
import textwrap

import pytest

from kinetic_sdk.plugin import (
    PluginLoader,
    PluginLoadError,
    PluginManifest,
    PluginVetError,
)
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.tool.base import Tool

from ._helpers import write_plugin


def load_directory_plugin(tmp_path, name, modules, entry_point="plugin:make_tools", audit=None):
    directory = write_plugin(tmp_path, name, entry_point=entry_point, modules=modules)
    manifest = PluginManifest.from_directory(directory)
    loader = PluginLoader(audit_logger=audit)
    return loader, loader.load(manifest)


class TestSuccessfulLoad:
    def test_loads_tools_from_directory_plugin(self, tmp_path):
        _, tools = load_directory_plugin(tmp_path, "hello", None)
        assert len(tools) == 1
        assert isinstance(tools[0], Tool)
        assert tools[0].name == "hello"
        assert tools[0].execute(message="hi").output == "hello:hi"

    def test_factory_manifest_hop_supported(self, tmp_path):
        # A factory may return a PluginManifest pointing at the real factory.
        modules = {
            "plugin.py": textwrap.dedent(
                """\
                from kinetic_sdk.plugin.manifest import PluginManifest

                def make_tools():
                    return PluginManifest(
                        name="hop",
                        version=None,
                        entry_point="real:build",
                        source="directory",
                        declared_capabilities=frozenset({"tool"}),
                        directory=__import__("pathlib").Path(__file__).parent.as_posix(),
                    )
                """
            ),
            "real.py": textwrap.dedent(
                """\
                from kinetic_sdk.tool.base import Tool, ToolResult

                class RealTool(Tool):
                    name = "real"
                    description = "d"
                    parameters = {"type": "object", "properties": {}}

                    def execute(self):
                        return ToolResult(output="real")

                def build():
                    return [RealTool()]
                """
            ),
        }
        _, tools = load_directory_plugin(tmp_path, "hop", modules)
        assert [t.name for t in tools] == ["real"]

    def test_package_entry_module_with_relative_import(self, tmp_path):
        modules = {
            "pkg/__init__.py": "",
            "pkg/tools.py": textwrap.dedent(
                """\
                from kinetic_sdk.tool.base import Tool, ToolResult
                from .helpers import SUFFIX

                class PkgTool(Tool):
                    name = "pkg-tool"
                    description = "d"
                    parameters = {"type": "object", "properties": {}}

                    def execute(self):
                        return ToolResult(output="pkg" + SUFFIX)

                def build():
                    return [PkgTool()]
                """
            ),
            "pkg/helpers.py": 'SUFFIX = "!"\n',
        }
        _, tools = load_directory_plugin(
            tmp_path, "pkgplug", modules, entry_point="pkg.tools:build"
        )
        assert tools[0].execute().output == "pkg!"

    def test_modules_registered_under_prefixed_names(self, tmp_path):
        load_directory_plugin(tmp_path, "hello", None)
        assert "kinetic_plugin_hello.plugin" in sys.modules
        assert "plugin" not in sys.modules  # never shadows real top-level names


class TestVetRejection:
    def test_critical_plugin_raises_and_is_never_imported(self, tmp_path):
        marker = tmp_path / "ran.marker"
        modules = {
            "plugin.py": (
                "import os\n"
                'os.system("echo hi")\n'
                f"open({str(marker)!r}, 'w').write('ran')\n"
                "def make_tools():\n    return []\n"
            )
        }
        directory = write_plugin(tmp_path, "evil", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        loader = PluginLoader()
        with pytest.raises(PluginVetError, match="evil"):
            loader.load(manifest)
        assert not marker.exists()  # import never happened
        assert "kinetic_plugin_evil.plugin" not in sys.modules

    def test_warning_only_plugin_still_loads(self, tmp_path):
        modules = {
            "plugin.py": (
                "import socket  # noqa: F401 - flagged as warning, not blocked\n"
                "def make_tools():\n    return []\n"
            )
        }
        _, tools = load_directory_plugin(tmp_path, "warner", modules)
        assert tools == []


class TestLoadFailures:
    def test_factory_exception_wrapped_in_load_error(self, tmp_path):
        modules = {
            "plugin.py": "def make_tools():\n    raise RuntimeError('boom')\n"
        }
        directory = write_plugin(tmp_path, "broken", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="broken") as exc_info:
            PluginLoader().load(manifest)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_import_error_wrapped_in_load_error(self, tmp_path):
        modules = {"plugin.py": "import no_such_module_xyz  # noqa: F401\n"}
        directory = write_plugin(tmp_path, "badimport", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="badimport"):
            PluginLoader().load(manifest)
        # Failed imports must not poison sys.modules for later attempts.
        assert "kinetic_plugin_badimport.plugin" not in sys.modules

    def test_non_list_factory_result_rejected(self, tmp_path):
        modules = {"plugin.py": "def make_tools():\n    return 'not-a-list'\n"}
        directory = write_plugin(tmp_path, "wrongtype", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="list"):
            PluginLoader().load(manifest)

    def test_non_tool_list_entry_rejected(self, tmp_path):
        modules = {"plugin.py": "def make_tools():\n    return [object()]\n"}
        directory = write_plugin(tmp_path, "nottool", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="non-Tool"):
            PluginLoader().load(manifest)

    def test_missing_factory_callable_rejected(self, tmp_path):
        modules = {"plugin.py": "TOOLS = []\n"}
        directory = write_plugin(tmp_path, "nofactory", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="not a callable"):
            PluginLoader().load(manifest)

    def test_unsupported_capability_rejected_before_scan(self, tmp_path):
        directory = write_plugin(tmp_path, "hooker", capabilities="hook")
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="hook"):
            PluginLoader().load(manifest)

    def test_entry_point_plugin_loads(self, tmp_path, monkeypatch):
        pkg = tmp_path / "eploaderpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "tools.py").write_text(
            textwrap.dedent(
                """\
                from kinetic_sdk.tool.base import Tool, ToolResult

                class EpTool(Tool):
                    name = "ep-tool"
                    description = "d"
                    parameters = {"type": "object", "properties": {}}

                    def execute(self):
                        return ToolResult(output="ep")

                def build():
                    return [EpTool()]
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "eploaderpkg", raising=False)
        manifest = PluginManifest.from_entry_point("ep", "eploaderpkg.tools:build")
        tools = PluginLoader().load(manifest)
        assert [t.name for t in tools] == ["ep-tool"]
        # Loading entry-point plugins imports them for real; clean up so
        # other tests are not affected by this test's import.
        sys.modules.pop("eploaderpkg", None)
        sys.modules.pop("eploaderpkg.tools", None)


class TestAuditLog:
    def entries(self, audit):
        return [e for e in audit.entries if e["event"] == "plugin_load"]

    def test_successful_load_audited(self, tmp_path):
        audit = InMemoryAuditLogger()
        load_directory_plugin(tmp_path, "hello", None, audit=audit)
        entries = self.entries(audit)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["plugin"] == "hello"
        assert entry["source"] == "directory"
        assert entry["outcome"] == "loaded"
        assert entry["tools"] == ["hello"]
        assert entry["scan"]["clean"] is True
        assert entry["scan"]["files_scanned"]

    def test_vet_rejection_audited(self, tmp_path):
        audit = InMemoryAuditLogger()
        modules = {"plugin.py": 'import os\nos.system("ls")\n'}
        directory = write_plugin(tmp_path, "evil", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginVetError):
            PluginLoader(audit_logger=audit).load(manifest)
        entries = self.entries(audit)
        assert len(entries) == 1
        assert entries[0]["outcome"] == "rejected"
        assert entries[0]["scan"]["clean"] is False
        categories = {f["category"] for f in entries[0]["scan"]["flags"]}
        assert "process_spawn" in categories

    def test_load_failure_audited(self, tmp_path):
        audit = InMemoryAuditLogger()
        modules = {"plugin.py": "def make_tools():\n    raise RuntimeError('x')\n"}
        directory = write_plugin(tmp_path, "broken", modules=modules)
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError):
            PluginLoader(audit_logger=audit).load(manifest)
        entries = self.entries(audit)
        assert entries[0]["outcome"] == "failed"
        assert "RuntimeError" in entries[0]["error"]

    def test_warnings_recorded_in_audit(self, tmp_path):
        audit = InMemoryAuditLogger()
        modules = {"plugin.py": "import socket\ndef make_tools():\n    return []\n"}
        load_directory_plugin(tmp_path, "warner", modules, audit=audit)
        entry = self.entries(audit)[0]
        assert entry["outcome"] == "loaded"
        assert any(
            f["category"] == "direct_network" for f in entry["scan"]["flags"]
        )
