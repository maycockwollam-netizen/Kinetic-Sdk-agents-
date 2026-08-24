"""The plugin loader: scan first, import second, audit everything.

Load order for every plugin is fixed and non-skippable:

1. :func:`~kinetic_sdk.plugin.scanner.scan_plugin` runs on the plugin's
   source. ``clean == False`` raises :class:`PluginVetError` and the plugin
   is NEVER imported. (A passing scan is a tripwire result, not a safety
   proof — see :mod:`kinetic_sdk.plugin.scanner`.)
2. Only then is the module imported and the declared factory called. The
   factory must return ``list[Tool]`` exactly — no coercion, no guessing.
3. EVERY load attempt, pass or fail, is written to the
   :class:`~kinetic_sdk.security.audit.AuditLogger`: plugin name, source,
   scan outcome (including non-blocking warnings), and the registered tool
   names. This is the only evidence trail for "which plugin brought tool X
   into the system".
4. Any exception during import/factory execution is wrapped in
   :class:`PluginLoadError` naming the plugin — one broken plugin never
   takes down the loading of the others.

Tools returned by a plugin are ordinary :class:`~kinetic_sdk.tool.base.Tool`
objects: they enter an agent's tool set exactly like internal or MCP tools
and are subject to ``permission_policy`` at execution time. There is no
privileged path.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from kinetic_sdk.plugin.exceptions import (
    PluginLoadError,
    PluginVetError,
)
from kinetic_sdk.plugin.manifest import PluginManifest
from kinetic_sdk.plugin.scanner import ScanResult, scan_plugin
from kinetic_sdk.security.audit import AuditLogger, InMemoryAuditLogger
from kinetic_sdk.tool.base import Tool

logger = logging.getLogger(__name__)

#: Audit event type used for every load attempt.
AUDIT_EVENT_PLUGIN_LOAD = "plugin_load"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PluginLoader:
    """Loads one plugin at a time: scan → import → factory → validate.

    Args:
        audit_logger: Sink for load audit entries. Defaults to an
            :class:`InMemoryAuditLogger` so loading is never unaudited;
            pass a shared logger (e.g. the agent's JSONL logger) to keep
            one trail.
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self._audit = audit_logger if audit_logger is not None else InMemoryAuditLogger()

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    # --- main entry ----------------------------------------------------------

    def load(self, manifest: PluginManifest) -> list[Tool]:
        """Scan, import and instantiate *manifest*'s plugin.

        Raises:
            PluginVetError: the static scanner found a critical pattern.
                The plugin was NOT imported.
            PluginLoadError: anything else went wrong (module missing,
                import/factory raised, factory returned the wrong type,
                unsupported capability). The original exception, if any, is
                chained as ``__cause__``.
        """
        unsupported = manifest.declared_capabilities - {"tool"}
        if unsupported:
            error = (
                f"capabilities {sorted(unsupported)} are declared but not "
                "supported by this SDK version (only 'tool' is implemented)"
            )
            self._audit_load(manifest, None, outcome="failed", error=error)
            raise PluginLoadError(f"plugin {manifest.name!r}: {error}")

        result = scan_plugin(manifest)
        if not result.clean:
            categories = sorted(
                {f.category for f in result.flags if f.severity == "critical"}
            )
            self._audit_load(manifest, result, outcome="rejected", error=str(categories))
            raise PluginVetError(
                f"plugin {manifest.name!r} failed static scanning: critical "
                f"findings in categories {categories} — the plugin was NOT "
                "imported. (A passing scan is a tripwire result, not a "
                "sandbox guarantee.)"
            )

        try:
            tools = self._import_and_build(manifest)
        except PluginLoadError as exc:
            self._audit_load(manifest, result, outcome="failed", error=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not kill the rest
            self._audit_load(
                manifest, result, outcome="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise PluginLoadError(
                f"plugin {manifest.name!r} raised during import/factory "
                f"execution: {type(exc).__name__}: {exc}"
            ) from exc

        self._validate_tools(manifest, tools)
        self._audit_load(
            manifest, result, outcome="loaded", tools=[tool.name for tool in tools]
        )
        return tools

    # --- import machinery ------------------------------------------------------

    def _import_and_build(self, manifest: PluginManifest) -> list[Tool]:
        module = self._import_module(manifest)
        produced = self._call_factory(manifest, module, manifest.factory_name)
        if isinstance(produced, PluginManifest):
            # Entry-point plugins may act as manifest providers: their entry
            # point yields the REAL manifest, whose factory then builds the
            # tools. One hop only — a manifest returned by that factory is
            # not followed further.
            if produced.source != manifest.source:
                produced = PluginManifest(
                    name=manifest.name,
                    version=produced.version or manifest.version,
                    entry_point=produced.entry_point,
                    source=manifest.source,
                    declared_capabilities=manifest.declared_capabilities,
                    directory=manifest.directory,
                )
            inner_module = self._import_module(produced)
            produced = self._call_factory(produced, inner_module, produced.factory_name)
        if not isinstance(produced, list):
            raise PluginLoadError(
                f"plugin {manifest.name!r}: factory {manifest.entry_point!r} "
                f"returned {type(produced).__name__}, expected list[Tool]"
            )
        return produced

    def _import_module(self, manifest: PluginManifest) -> ModuleType:
        if manifest.source == "directory":
            return self._import_directory_module(manifest)
        return importlib.import_module(manifest.module_spec)

    @staticmethod
    def _call_factory(
        manifest: PluginManifest, module: ModuleType, factory_name: str
    ) -> Any:
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise PluginLoadError(
                f"plugin {manifest.name!r}: {factory_name!r} is not a callable "
                f"in module {module.__name__!r}"
            )
        return factory()

    @staticmethod
    def _import_directory_module(manifest: PluginManifest) -> ModuleType:
        """Import ``manifest.module_spec`` from the plugin directory.

        Modules are registered under a ``kinetic_plugin_<name>`` prefix so
        they can never shadow real installed packages in ``sys.modules``;
        parent packages are materialised along the way so intra-plugin
        imports (``from .helpers import x`` / ``from pkg.mod import y``)
        work without touching ``sys.path``.
        """
        directory = Path(manifest.directory or "")
        parts = manifest.module_spec.split(".")
        prefix = f"kinetic_plugin_{manifest.name.replace('-', '_')}"
        inserted: list[str] = []
        parent: ModuleType | None = None
        current = directory
        try:
            for index, part in enumerate(parts):
                full_name = f"{prefix}." + ".".join(parts[: index + 1])
                is_last = index == len(parts) - 1
                package_init = current / part / "__init__.py"
                module_file = current / f"{part}.py"
                if package_init.is_file():
                    spec = importlib.util.spec_from_file_location(
                        full_name,
                        package_init,
                        submodule_search_locations=[str(current / part)],
                    )
                    current = current / part
                elif module_file.is_file():
                    if not is_last:
                        raise PluginLoadError(
                            f"plugin {manifest.name!r}: {part!r} in "
                            f"{manifest.module_spec!r} is a module, not a package"
                        )
                    spec = importlib.util.spec_from_file_location(full_name, module_file)
                else:
                    raise PluginLoadError(
                        f"plugin {manifest.name!r}: cannot locate module "
                        f"{manifest.module_spec!r} under {directory}"
                    )
                if spec is None or spec.loader is None:
                    raise PluginLoadError(
                        f"plugin {manifest.name!r}: cannot load spec for "
                        f"{full_name!r} ({module_file})"
                    )
                module = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = module
                inserted.append(full_name)
                if parent is not None:
                    setattr(parent, part, module)
                spec.loader.exec_module(module)
                parent = module
        except Exception:
            for name in inserted:
                sys.modules.pop(name, None)
            raise
        assert parent is not None
        return parent

    # --- validation & audit ---------------------------------------------------

    @staticmethod
    def _validate_tools(manifest: PluginManifest, tools: list[Any]) -> None:
        for tool in tools:
            if not isinstance(tool, Tool):
                raise PluginLoadError(
                    f"plugin {manifest.name!r}: factory returned a non-Tool "
                    f"object ({type(tool).__name__}); every entry must be a "
                    "kinetic_sdk.tool.base.Tool"
                )

    def _audit_load(
        self,
        manifest: PluginManifest,
        result: ScanResult | None,
        outcome: str,
        tools: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "plugin": manifest.name,
            "version": manifest.version,
            "source": manifest.source,
            "entry_point": manifest.entry_point,
            "outcome": outcome,
        }
        if result is not None:
            fields["scan"] = {
                "clean": result.clean,
                "files_scanned": result.files_scanned,
                "flags": [
                    {
                        "severity": flag.severity,
                        "category": flag.category,
                        "message": flag.message,
                        "location": flag.location,
                    }
                    for flag in result.flags
                ],
            }
        if tools is not None:
            fields["tools"] = tools
        if error is not None:
            fields["error"] = error
        self._audit.log_event(
            AUDIT_EVENT_PLUGIN_LOAD, manifest.name, _utcnow(), **fields
        )
