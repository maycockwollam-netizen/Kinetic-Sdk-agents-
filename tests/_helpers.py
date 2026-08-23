"""Shared test helpers: the public mock LLM re-exported plus extra fakes.

``MockLLM``/``text_response``/``tool_response`` are aliases of the public
utilities in :mod:`kinetic_sdk.testing` (kept under the old names so existing
tests stay unchanged). ``EchoTool``/``FailingTool`` are fixtures specific to
this suite and remain test-only. ``write_skill``/``make_zip`` build skill
fixtures for the ``skills/`` tests.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.testing import mocks as _mocks
from kinetic_sdk.tool.base import Tool, ToolResult

#: Backwards-compatible aliases: the SDK's own tests predate ``testing/``.
MockLLM = _mocks.MockLLMClient
text_response = _mocks.text_response
tool_response = _mocks.tool_response


class LoopLLM(LLMClient):
    """An LLM that requests the SAME tool call on every turn, forever.

    Runaway fixture for the subagent tests: a model that never stops
    calling one tool with identical arguments is exactly what the budget /
    circuit-breaker guardrails must contain. ``calls`` counts total chat
    invocations (shared across every agent inheriting this instance).
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        model: str = "loop-model",
    ) -> None:
        self.model = model
        self.tool_name = tool_name
        self.arguments = dict(arguments or {})
        self.calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        return tool_response(
            f"{self.tool_name}-{self.calls}", self.tool_name, dict(self.arguments)
        )


class EchoTool(Tool):
    """A trivial tool that echoes its input plus an optional prefix."""

    name = "echo"
    description = "Echo back the provided message, optionally with a prefix."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Text to echo."},
            "prefix": {"type": "string", "description": "Optional prefix.", "default": ""},
        },
        "required": ["message"],
    }

    def __init__(self, prefix: str = "") -> None:
        self._default_prefix = prefix

    def execute(self, message: str, prefix: str | None = None) -> ToolResult:
        p = prefix if prefix is not None else self._default_prefix
        return ToolResult(output=f"{p}{message}")


class FailingTool(Tool):
    """A tool that always raises, to exercise the agent's error handling."""

    name = "boom"
    description = "Always raises an exception."
    parameters = {"type": "object", "properties": {}}

    def execute(self, **params: Any) -> ToolResult:
        raise RuntimeError("kaboom")


def write_skill(
    base: Path,
    name: str,
    *,
    description: str = "A demo skill.",
    body: str = "Demo body.",
    resources: dict[str, str] | None = None,
    frontmatter_extras: dict[str, str] | None = None,
    frontmatter_name: str | None = None,
) -> Path:
    """Create a skill directory (SKILL.md + optional resources) under *base*.

    ``frontmatter_name`` overrides the ``name`` written into the frontmatter
    so tests can exercise the name-must-match-directory rule.
    """
    directory = base / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {frontmatter_name if frontmatter_name is not None else name}",
        f"description: {description}",
    ]
    for key, value in (frontmatter_extras or {}).items():
        lines.append(f"{key}: {value}")
    lines += ["---", "", body]
    (directory / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    for relative, content in (resources or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return directory


def make_zip(path: Path, entries: dict[str, str | None]) -> Path:
    """Write a zip at *path*; a ``None`` content marks a directory entry."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            if content is None:
                archive.writestr(name if name.endswith("/") else name + "/", "")
            else:
                archive.writestr(name, content)
    return path


def skill_zip_entries(
    name: str,
    *,
    description: str = "A zipped skill.",
    body: str = "Zipped body.",
    resources: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """Build the entries dict for a single valid skill inside a zip."""
    skill_md = (
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    )
    entries: dict[str, str | None] = {f"{name}/SKILL.md": skill_md}
    for relative, content in (resources or {}).items():
        entries[f"{name}/{relative}"] = content
    return entries


#: A minimal but fully valid plugin factory module used by plugin tests.
PLUGIN_CLEAN_MODULE = '''from kinetic_sdk.tool.base import Tool, ToolResult


class {class_name}(Tool):
    name = "{tool_name}"
    description = "Echoes its input (plugin fixture)."
    parameters = {{"type": "object", "properties": {{"message": {{"type": "string"}}}}}}

    def execute(self, message: str = "") -> ToolResult:
        return ToolResult(output=f"{tool_name}:{{message}}")


def make_tools():
    return [{class_name}()]
'''


def write_plugin(
    base: Path,
    name: str,
    *,
    entry_point: str = "plugin:make_tools",
    version: str | None = "0.1",
    capabilities: str = "tool",
    modules: dict[str, str] | None = None,
    frontmatter_name: str | None = None,
) -> Path:
    """Create a plugin directory (PLUGIN.md + Python files) under *base*.

    ``modules`` maps plugin-relative file paths to source text; when omitted
    a single clean ``plugin.py`` exposing ``make_tools`` (matching the
    default *entry_point*) is written. ``frontmatter_name`` overrides the
    ``name`` written into the frontmatter to exercise the
    name-must-match-directory rule.
    """
    directory = base / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {frontmatter_name if frontmatter_name is not None else name}",
        f"entry_point: {entry_point}",
        f"capabilities: {capabilities}",
    ]
    if version is not None:
        lines.insert(2, f"version: {version}")
    lines += ["---", "", f"# {name} plugin fixture"]
    (directory / "PLUGIN.md").write_text("\n".join(lines), encoding="utf-8")
    if modules is None:
        class_name = "".join(part.capitalize() for part in name.split("-")) + "Tool"
        modules = {
            "plugin.py": PLUGIN_CLEAN_MODULE.format(
                class_name=class_name, tool_name=name
            )
        }
    for relative, content in modules.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return directory


class FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint (name/value/dist)."""

    def __init__(self, name: str, value: str, version: str | None = None) -> None:
        self.name = name
        self.value = value
        if version is not None:
            self.dist = type("FakeDist", (), {"version": version})()
