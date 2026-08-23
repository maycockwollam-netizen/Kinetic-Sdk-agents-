"""Tests for kinetic_sdk.plugin.registry + end-to-end integration with the
agent loop (plugin tools must pass through permission_policy like any other
tool).
"""
from __future__ import annotations

import textwrap

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.plugin import (
    PluginLoadError,
    PluginManifestError,
    PluginRegistry,
    PluginVetError,
)
from kinetic_sdk.security import AllowListPolicy, PermissivePolicy

from ._helpers import (
    FakeEntryPoint,
    MockLLM,
    text_response,
    tool_response,
    write_plugin,
)

EVIL_MODULES = {"plugin.py": 'import os\nos.system("ls")\ndef make_tools():\n    return []\n'}
BROKEN_MODULES = {"plugin.py": "def make_tools():\n    raise RuntimeError('boom')\n"}


def tool_module(tool_name: str) -> dict[str, str]:
    return {
        "plugin.py": textwrap.dedent(
            f"""\
            from kinetic_sdk.tool.base import Tool, ToolResult

            class T(Tool):
                name = "{tool_name}"
                description = "d"
                parameters = {{"type": "object", "properties": {{"message": {{"type": "string"}}}}}}

                def execute(self, message=""):
                    return ToolResult(output="{tool_name}:" + message)

            def make_tools():
                return [T()]
            """
        )
    }


class TestLoadAll:
    def test_skip_mode_loads_good_skips_bad(self, tmp_path, caplog):
        write_plugin(tmp_path, "good")
        write_plugin(tmp_path, "evil", modules=EVIL_MODULES)
        write_plugin(tmp_path, "broken", modules=BROKEN_MODULES)
        registry = PluginRegistry(directory=tmp_path)
        with caplog.at_level("WARNING"):
            tools = registry.load_all()
        assert [t.name for t in tools] == ["good"]
        skipped = " ".join(r.message for r in caplog.records)
        assert "evil" in skipped and "broken" in skipped

    def test_raise_mode_fails_fast_on_vet_error(self, tmp_path):
        write_plugin(tmp_path, "evil", modules=EVIL_MODULES)
        registry = PluginRegistry(directory=tmp_path)
        with pytest.raises(PluginVetError):
            registry.load_all(on_error="raise")

    def test_raise_mode_fails_fast_on_load_error(self, tmp_path):
        write_plugin(tmp_path, "broken", modules=BROKEN_MODULES)
        registry = PluginRegistry(directory=tmp_path)
        with pytest.raises(PluginLoadError):
            registry.load_all(on_error="raise")

    def test_invalid_on_error_rejected(self, tmp_path):
        registry = PluginRegistry(directory=tmp_path)
        with pytest.raises(ValueError):
            registry.load_all(on_error="explode")

    def test_empty_when_no_plugins(self):
        registry = PluginRegistry(entry_points_provider=lambda: [])
        assert registry.load_all() == []

    def test_merges_entry_point_and_directory_plugins(self, tmp_path, monkeypatch):
        write_plugin(tmp_path, "dirplug")
        pkg = tmp_path / "eppkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod.py").write_text(tool_module("ep-plug")["plugin.py"], encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        registry = PluginRegistry(
            directory=tmp_path,
            entry_points_provider=lambda: [FakeEntryPoint("ep-plug", "eppkg.mod:make_tools")],
        )
        names = sorted(t.name for t in registry.load_all())
        assert names == ["dirplug", "ep-plug"]


class TestCollisions:
    def test_tool_name_collision_between_plugins_raises(self, tmp_path):
        write_plugin(tmp_path, "plug-a", modules=tool_module("shared"))
        write_plugin(tmp_path, "plug-b", modules=tool_module("shared"))
        registry = PluginRegistry(directory=tmp_path)
        with pytest.raises(PluginLoadError, match="shared"):
            registry.load_all()

    def test_reserved_names_are_protected(self, tmp_path):
        write_plugin(tmp_path, "plug", modules=tool_module("git"))
        registry = PluginRegistry(directory=tmp_path, reserved_names=["git"])
        with pytest.raises(PluginLoadError, match="reserved"):
            registry.load_all()

    def test_duplicate_plugin_name_across_sources_raises(self, tmp_path):
        write_plugin(tmp_path, "clash")
        registry = PluginRegistry(
            directory=tmp_path,
            entry_points_provider=lambda: [FakeEntryPoint("clash", "pkg:build")],
        )
        with pytest.raises(PluginManifestError, match="clash"):
            registry.load_all()


class TestAgentIntegration:
    def _run_agent(self, tmp_path, policy):
        write_plugin(tmp_path, "plug", modules=tool_module("plug-tool"))
        tools = PluginRegistry(directory=tmp_path).load_all()
        llm = MockLLM(
            [
                tool_response("call-1", "plug-tool", {"message": "hi"}),
                text_response("done"),
            ]
        )
        agent = Agent(llm=llm, tools=tools, permission_policy=policy)
        return llm, agent.run("call the plugin tool")

    def test_plugin_tool_executes_under_permissive_policy(self, tmp_path):
        llm, result = self._run_agent(tmp_path, PermissivePolicy())
        assert result == "done"
        # The tool result fed back to the model is the plugin tool's output.
        second_call_messages = llm.calls[1]["messages"]
        tool_results = [
            block
            for message in second_call_messages
            for block in (
                message["content"] if isinstance(message.get("content"), list) else []
            )
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert any("plug-tool:hi" in str(block) for block in tool_results)

    def test_plugin_tool_denied_by_policy_like_any_tool(self, tmp_path):
        # An empty AllowListPolicy denies EVERYTHING — a plugin tool must
        # get no bypass relative to internal/MCP tools.
        llm, result = self._run_agent(tmp_path, AllowListPolicy())
        assert result == "done"  # agent still finishes; the CALL is denied
        second_call_messages = llm.calls[1]["messages"]
        tool_results = [
            block
            for message in second_call_messages
            for block in (
                message["content"] if isinstance(message.get("content"), list) else []
            )
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert tool_results
        assert all("plug-tool:hi" not in str(block) for block in tool_results)

    def test_plugin_tool_denied_emits_permission_denied_event(self, tmp_path):
        from kinetic_sdk.observability import InMemoryObservabilityLogger

        write_plugin(tmp_path, "plug", modules=tool_module("plug-tool"))
        tools = PluginRegistry(directory=tmp_path).load_all()
        llm = MockLLM(
            [
                tool_response("call-1", "plug-tool", {"message": "hi"}),
                text_response("done"),
            ]
        )
        obs = InMemoryObservabilityLogger()
        agent = Agent(
            llm=llm,
            tools=tools,
            permission_policy=AllowListPolicy(),
            observability_logger=obs,
        )
        agent.run("call the plugin tool")
        denied = obs.get_events("security.permission_denied")
        assert len(denied) == 1
        assert denied[0]["payload"]["name"] == "plug-tool"

    def test_plugin_tool_allowed_by_explicit_allowlist(self, tmp_path):
        from kinetic_sdk.observability import InMemoryObservabilityLogger

        write_plugin(tmp_path, "plug", modules=tool_module("plug-tool"))
        tools = PluginRegistry(directory=tmp_path).load_all()
        llm = MockLLM(
            [
                tool_response("call-1", "plug-tool", {"message": "hi"}),
                text_response("done"),
            ]
        )
        obs = InMemoryObservabilityLogger()
        agent = Agent(
            llm=llm,
            tools=tools,
            permission_policy=AllowListPolicy(always_allow=["plug-tool"]),
            observability_logger=obs,
        )
        agent.run("call the plugin tool")
        finished = obs.get_events("agent.tool_call_finished")
        assert len(finished) == 1
        assert finished[0]["payload"]["is_error"] is False
        assert not obs.get_events("security.permission_denied")
