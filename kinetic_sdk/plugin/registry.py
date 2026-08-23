"""The plugin registry: discovery + scanning + loading behind one API.

Mirrors :class:`~kinetic_sdk.skills.registry.SkillRegistry`'s role for
skills, but for executable plugins::

    registry = PluginRegistry(directory="~/.kinetic/plugins")
    tools = registry.load_all()          # skip-and-log broken plugins
    agent = Agent(llm=..., tools=[*tools, GitTool()], ...)

**Tool-name collisions are REJECTED, never silently resolved.** MCP tools
are prefixed (``server.tool``) because two independent servers legitimately
export the same generic name; plugin tools run in-process under names the
plugin author chose freely, so a collision there means either a packaging
mistake or an attempt to shadow another tool in the model's eyes. Renaming
a plugin's tool behind its back would change what the model is told to
call, so :meth:`PluginRegistry.load_all` raises instead — fix the plugin,
don't paper over it. Optional *reserved_names* lets callers protect their
internal/MCP tool names from shadowing by plugins.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Literal

from kinetic_sdk.plugin.discovery import EntryPointsProvider, discover_plugins
from kinetic_sdk.plugin.exceptions import PluginLoadError, PluginManifestError
from kinetic_sdk.plugin.loader import PluginLoader
from kinetic_sdk.plugin.manifest import PluginManifest
from kinetic_sdk.security.audit import AuditLogger
from kinetic_sdk.tool.base import Tool

logger = logging.getLogger(__name__)

OnErrorMode = Literal["raise", "skip"]


class PluginRegistry:
    """High-level plugin API: one call discovers, scans, loads, merges.

    Args:
        directory: Directory-convention plugin root (e.g.
            ``~/.kinetic/plugins``); ``None`` disables that source.
        entry_points_provider: Replacement for
            ``importlib.metadata.entry_points`` (tests). ``None`` uses the
            real installed-package metadata.
        audit_logger: Shared audit sink for load entries (pass the agent's
            logger to keep one trail).
        loader: Injectable :class:`PluginLoader` (tests / custom policies).
            When given, *audit_logger* is ignored — the loader owns its sink.
        reserved_names: Tool names plugins may NOT claim (typically the
            caller's internal + MCP tool names). A plugin tool colliding
            with one raises :class:`PluginLoadError` at merge time.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        entry_points_provider: EntryPointsProvider | None = None,
        audit_logger: AuditLogger | None = None,
        loader: PluginLoader | None = None,
        reserved_names: Iterable[str] = (),
    ) -> None:
        self._directory = directory
        self._entry_points_provider = entry_points_provider
        self._loader = loader if loader is not None else PluginLoader(audit_logger)
        self._reserved_names = frozenset(reserved_names)

    def discover(self) -> list[PluginManifest]:
        """Discover all candidate plugins (metadata only, no imports)."""
        return discover_plugins(
            directory=self._directory,
            entry_points_provider=self._entry_points_provider,
        )

    def load_all(self, on_error: OnErrorMode = "skip") -> list[Tool]:
        """Discover, scan and load every plugin; return the merged tools.

        Args:
            on_error: ``"skip"`` (default) — a plugin that fails vetting or
                loading is logged and skipped, so one bad plugin never
                blocks agent startup. ``"raise"`` — the first failure
                propagates, for strict environments that prefer fail-fast.

        Raises:
            ValueError: *on_error* is not ``"raise"`` or ``"skip"``.
            PluginManifestError: duplicate plugin names across sources
                (always raised — discovery ambiguity, not a load error).
            PluginLoadError: a plugin tool's name collides with another
                plugin's tool or with a reserved name (always raised —
                see the module docstring for why collisions are rejected
                rather than renamed).
        """
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
        tools: list[Tool] = []
        owners: dict[str, str] = {}
        for manifest in self.discover():
            try:
                loaded = self._loader.load(manifest)
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning(
                    "skipping plugin %r (%s): %s",
                    manifest.name,
                    manifest.source,
                    exc,
                )
                continue
            for tool in loaded:
                owner = owners.get(tool.name)
                if owner is not None:
                    raise PluginLoadError(
                        f"tool name {tool.name!r} is claimed by both plugin "
                        f"{owner!r} and plugin {manifest.name!r} — collisions "
                        "are rejected, not renamed (see plugin/registry.py "
                        "docstring)"
                    )
                if tool.name in self._reserved_names:
                    raise PluginLoadError(
                        f"plugin {manifest.name!r} declares tool {tool.name!r} "
                        "which is reserved by an internal/MCP tool — plugin "
                        "tools may not shadow existing tools"
                    )
                owners[tool.name] = manifest.name
                tools.append(tool)
        return tools
