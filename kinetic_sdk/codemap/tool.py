"""Agent-facing, plain-text views over the cached static codebase map."""

from __future__ import annotations

from typing import Any

from kinetic_sdk.codemap.cache import CodebaseMapCache
from kinetic_sdk.codemap.models import CodebaseMap
from kinetic_sdk.tool.base import Tool, ToolResult


class CodebaseMapTool(Tool):
    """Query import-level dependencies before editing unfamiliar code.

    Call ``impact_of`` before changing a module you did not just read to see
    what else might depend on it. The map is cached, so repeated calls are
    cheap. It is import-level only, not a call graph: dynamic imports and
    string-based plugin loading can hide coupling, so read affected files and
    treat missing edges as unknown rather than proof of isolation.
    """

    name = "codebase_map"
    description = (
        "Before editing an unfamiliar module, call impact_of to see what might depend "
        "on it. This cached, cheap-to-repeat map covers imports only, not a call "
        "graph: dynamic imports and string-based loading can hide coupling, so a "
        "missing edge is unknown rather than proof that files are independent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "enum": ["overview", "imports_of", "importers_of", "impact_of"], "description": "Map view to return."},
            "module": {"type": "string", "description": "Dotted module name, required except for overview."},
            "max_depth": {"type": "integer", "description": "Reverse-import hops for impact_of; omit for unlimited."},
        },
        "required": ["query"],
    }

    def __init__(self, root_path: str, *, cache: CodebaseMapCache | None = None) -> None:
        self._cache = cache or CodebaseMapCache(root_path)

    @staticmethod
    def _node_names(codebase_map: CodebaseMap) -> set[str]:
        return {node.module_name for node in codebase_map.modules}

    def _require_module(self, codebase_map: CodebaseMap, module: str | None) -> ToolResult | None:
        if not module:
            return ToolResult(error="module is required for this codebase_map query")
        names = self._node_names(codebase_map)
        if module not in names:
            needle = module.lower()
            suggestions = [name for name in sorted(names) if needle in name.lower() or name.lower() in needle]
            hint = f" Suggestions: {', '.join(suggestions[:5])}." if suggestions else ""
            return ToolResult(error=f"module {module!r} is not in the codebase map.{hint}")
        return None

    def execute(self, query: str, module: str | None = None, max_depth: int | None = None, **_: Any) -> ToolResult:  # type: ignore[override]
        """Render a compact map view; invalid query inputs are recoverable errors."""
        if query not in {"overview", "imports_of", "importers_of", "impact_of"}:
            return ToolResult(error=f"unknown codebase_map query {query!r}")
        if max_depth is not None and (not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0):
            return ToolResult(error="max_depth must be a non-negative integer or omitted")
        codebase_map = self._cache.get_or_build()
        if query == "overview":
            edges = sum(len(node.imports) for node in codebase_map.modules)
            lines = [f"Codebase map: {len(codebase_map.modules)} modules, {edges} import edges."]
            if codebase_map.skipped:
                lines.append("Skipped unparsable files: " + ", ".join(codebase_map.skipped))
            return ToolResult(output="\n".join(lines))
        error = self._require_module(codebase_map, module)
        if error is not None:
            return error
        assert module is not None
        if query == "imports_of":
            imports = codebase_map.imports_of(module)
            return ToolResult(output=f"{len(imports)} modules imported by {module}:" + ("\n" + "\n".join(f"- {item}" for item in imports) if imports else " none"))
        if query == "importers_of":
            importers = codebase_map.importers_of(module)
            return ToolResult(output=f"{len(importers)} modules import {module}:" + ("\n" + "\n".join(f"- {item}" for item in importers) if importers else " none"))
        importers = codebase_map.transitive_importers_of(module, max_depth=max_depth)
        depths: dict[str, int] = {module: 0}
        frontier = [module]
        while frontier:
            current = frontier.pop(0)
            for importer in codebase_map.importers_of(current):
                if importer not in depths:
                    depths[importer] = depths[current] + 1
                    frontier.append(importer)
        lines = [f"Impact of changing {module}: {len(importers)} downstream importers."]
        for item in importers:
            lines.append(f"{'  ' * (depths[item] - 1)}- hop {depths[item]}: {item}")
        return ToolResult(output="\n".join(lines))
