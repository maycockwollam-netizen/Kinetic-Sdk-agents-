"""Tests for kinetic_sdk.plugin.scanner — every pattern gets a positive
(bad code IS flagged) and a negative (similar legitimate code is NOT
flagged) case, plus whole-plugin integration through scan_plugin().
"""
from __future__ import annotations

import textwrap

import pytest

from kinetic_sdk.plugin import (
    PluginLoadError,
    PluginManifest,
    StaticPluginScanner,
    scan_plugin,
)

from ._helpers import write_plugin


def scan(source: str):
    return StaticPluginScanner().scan_source(textwrap.dedent(source), "plugin.py")


def categories(flags, severity=None):
    return sorted(
        {f.category for f in flags if severity is None or f.severity == severity}
    )


class TestCleanCode:
    def test_fully_clean_plugin_has_no_flags(self):
        flags = scan(
            '''\
            from kinetic_sdk.tool.base import Tool, ToolResult
            from kinetic_sdk.secret import SecretRegistry

            class MyTool(Tool):
                name = "mine"
                description = "d"
                parameters = {"type": "object", "properties": {}}

                def execute(self):
                    return ToolResult(output=SecretRegistry.__name__)

            def make_tools():
                return [MyTool()]
            '''
        )
        assert flags == []

    def test_unparseable_source_is_critical(self):
        flags = scan("def broken(:\n")
        assert categories(flags) == ["unparseable"]


class TestEvalExecCompile:
    @pytest.mark.parametrize("func", ["eval", "exec", "compile"])
    def test_dynamic_input_is_critical(self, func):
        flags = scan(f"result = {func}(user_input)")
        assert categories(flags, "critical") == ["code_execution"]

    @pytest.mark.parametrize("func", ["eval", "exec", "compile"])
    def test_literal_input_not_flagged(self, func):
        assert scan(f'{func}("1 + 1")') == []

    def test_keyword_input_flagged(self):
        flags = scan("eval(source=expression)")
        assert categories(flags, "critical") == ["code_execution"]


class TestSubprocessAndShell:
    @pytest.mark.parametrize(
        "call",
        [
            'os.system("ls")',
            'os.popen("ls")',
            'subprocess.Popen(["ls"])',
            'subprocess.run(["ls"])',
            'subprocess.call(["ls"])',
            'subprocess.check_output(["ls"])',
        ],
    )
    def test_direct_calls_are_critical(self, call):
        flags = scan(f"import os, subprocess\n{call}")
        assert "process_spawn" in categories(flags, "critical")

    def test_from_import_alias_is_caught(self):
        flags = scan('from subprocess import run as go\ngo(["ls"])')
        assert "process_spawn" in categories(flags, "critical")

    def test_import_alias_is_caught(self):
        flags = scan('import subprocess as sp\nsp.run(["ls"])')
        assert "process_spawn" in categories(flags, "critical")

    def test_git_tool_wrapper_not_flagged(self):
        # Using the SDK's controlled GitTool wrapper must NOT be blocked,
        # even though git itself shells out internally.
        flags = scan(
            '''\
            from kinetic_sdk.git import GitTool

            def make_tools():
                return [GitTool()]
            '''
        )
        assert flags == []

    def test_unrelated_run_method_not_flagged(self):
        flags = scan(
            '''\
            class Runner:
                def run(self):
                    return 1

            Runner().run()
            '''
        )
        assert flags == []


class TestOsEnviron:
    def test_read_is_critical(self):
        flags = scan('import os\ntoken = os.environ["TOKEN"]')
        assert "env_access" in categories(flags, "critical")

    def test_write_is_critical(self):
        flags = scan('import os\nos.environ["X"] = "1"')
        assert "env_access" in categories(flags, "critical")

    def test_method_access_is_critical(self):
        flags = scan('import os\nos.environ.get("X")')
        assert "env_access" in categories(flags, "critical")

    def test_from_import_is_critical(self):
        flags = scan('from os import environ\nenviron["X"]')
        assert "env_access" in categories(flags, "critical")

    def test_getattr_evasion_is_critical(self):
        flags = scan('import os\ngetattr(os, "environ")')
        assert "env_access" in categories(flags, "critical")

    def test_secret_registry_not_flagged(self):
        flags = scan(
            '''\
            from kinetic_sdk.secret import SecretRegistry

            def make_tools():
                return SecretRegistry()
            '''
        )
        assert flags == []

    def test_unrelated_environ_variable_not_flagged(self):
        flags = scan('environ = {"X": "1"}\nprint(environ["X"])')
        assert flags == []


class TestSecurityTamper:
    def test_assign_into_security_namespace_is_critical(self):
        flags = scan(
            '''\
            import kinetic_sdk.security.policy as policy
            policy.PermissionPolicy.check = lambda self, a, b: None
            '''
        )
        assert "security_tamper" in categories(flags, "critical")

    def test_assign_produces_exactly_one_flag(self):
        flags = scan(
            '''\
            import kinetic_sdk.security.policy as policy
            policy.PermissionPolicy.check = lambda self, a, b: None
            '''
        )
        tampers = [f for f in flags if f.category == "security_tamper"]
        assert len(tampers) == 1

    def test_setattr_into_security_namespace_is_critical(self):
        flags = scan(
            '''\
            from kinetic_sdk.security import policy
            setattr(policy, "AllowListPolicy", object)
            '''
        )
        assert "security_tamper" in categories(flags, "critical")

    def test_full_dotted_path_assignment_is_critical(self):
        flags = scan(
            '''\
            import kinetic_sdk.security.policy
            kinetic_sdk.security.policy.AllowListPolicy.check = None
            '''
        )
        assert "security_tamper" in categories(flags, "critical")

    def test_delete_from_security_namespace_is_critical(self):
        flags = scan(
            '''\
            import kinetic_sdk.security.policy as policy
            del policy.PermissionPolicy
            '''
        )
        assert "security_tamper" in categories(flags, "critical")

    def test_public_api_use_not_flagged(self):
        flags = scan(
            '''\
            from kinetic_sdk.security import AllowListPolicy

            POLICY = AllowListPolicy(always_allow=["git"])
            '''
        )
        assert flags == []

    def test_other_kinetic_sdk_modules_not_flagged(self):
        flags = scan(
            '''\
            import kinetic_sdk.tool.base as base
            print(base.Tool)
            '''
        )
        assert flags == []


class TestDynamicImport:
    def test_dunder_import_non_constant_is_critical(self):
        flags = scan("mod = __import__(module_name)")
        assert "dynamic_import" in categories(flags, "critical")

    def test_importlib_non_constant_is_critical(self):
        flags = scan("import importlib\nimportlib.import_module(name)")
        assert "dynamic_import" in categories(flags, "critical")

    def test_from_importlib_alias_is_critical(self):
        flags = scan("from importlib import import_module\nimport_module(name)")
        assert "dynamic_import" in categories(flags, "critical")

    def test_constant_module_name_not_flagged(self):
        flags = scan('import importlib\nimportlib.import_module("json")')
        assert flags == []
        flags = scan('__import__("json")')
        assert flags == []


class TestNetworkImports:
    @pytest.mark.parametrize(
        "statement",
        [
            "import socket",
            "import urllib.request",
            "from urllib import request",
            "import http.client",
            "import requests",
            "import httpx",
        ],
    )
    def test_network_import_is_warning(self, statement):
        flags = scan(statement)
        assert categories(flags) == ["direct_network"]
        assert all(f.severity == "warning" for f in flags)

    def test_no_network_import_no_flag(self):
        assert scan("import json\nimport pathlib") == []


class TestOpenCalls:
    def test_non_literal_path_is_warning(self):
        flags = scan("open(user_path)")
        assert categories(flags) == ["dynamic_file_access"]

    def test_literal_path_not_flagged(self):
        assert scan('open("data/config.json")') == []

    def test_pathlib_usage_not_flagged(self):
        flags = scan(
            '''\
            from pathlib import Path
            Path("x.txt").read_text()
            '''
        )
        assert flags == []


class TestScanPluginIntegration:
    def manifest_for(self, tmp_path, name, modules, entry_point="plugin:make_tools"):
        directory = write_plugin(tmp_path, name, entry_point=entry_point, modules=modules)
        return PluginManifest.from_directory(directory)

    def test_clean_directory_plugin(self, tmp_path):
        manifest = self.manifest_for(
            tmp_path,
            "clean",
            {"plugin.py": "def make_tools():\n    return []\n"},
        )
        result = scan_plugin(manifest)
        assert result.clean
        assert result.flags == []
        assert result.files_scanned and result.files_scanned[0].endswith("plugin.py")

    def test_all_py_files_are_scanned_not_just_entry(self, tmp_path):
        manifest = self.manifest_for(
            tmp_path,
            "multi",
            {
                "plugin.py": "def make_tools():\n    return []\n",
                "helpers/evil.py": 'import os\nos.system("ls")\n',
            },
        )
        result = scan_plugin(manifest)
        assert not result.clean
        assert any("evil.py" in f.location for f in result.flags)

    def test_flag_locations_include_line_numbers(self, tmp_path):
        manifest = self.manifest_for(
            tmp_path,
            "located",
            {"plugin.py": 'import os\n\n\ntoken = os.environ["T"]\n'},
        )
        result = scan_plugin(manifest)
        assert any(f.location.endswith(":4") for f in result.flags)

    def test_directory_without_python_files_raises(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "PLUGIN.md").write_text(
            "---\nname: empty\nentry_point: plugin:make_tools\n---\n", encoding="utf-8"
        )
        manifest = PluginManifest.from_directory(directory)
        with pytest.raises(PluginLoadError, match="no Python files"):
            scan_plugin(manifest)

    def test_entry_point_source_located_without_import(self, tmp_path, monkeypatch):
        pkg = tmp_path / "epscannerpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "tools.py").write_text(
            "import socket\n\ndef build():\n    return []\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        manifest = PluginManifest.from_entry_point("ep", "epscannerpkg.tools:build")
        result = scan_plugin(manifest)
        assert result.clean  # socket is a warning, not a blocker
        assert categories(result.flags) == ["direct_network"]
        import sys

        assert "epscannerpkg" not in sys.modules  # scanning must not import

    def test_entry_point_module_not_found_raises(self):
        manifest = PluginManifest.from_entry_point("ghost", "no_such_mod_xyz:build")
        with pytest.raises(PluginLoadError, match="cannot locate module"):
            scan_plugin(manifest)
