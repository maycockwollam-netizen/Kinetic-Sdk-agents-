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

from kinetic_sdk.testing.mocks import (
    MockLLMClient,
    text_response,
    tool_response,
)
from kinetic_sdk.tool.base import Tool, ToolResult

#: Backwards-compatible alias: the SDK's own tests predate ``testing/``.
MockLLM = MockLLMClient


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
