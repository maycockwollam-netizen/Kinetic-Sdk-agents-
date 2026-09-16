"""Immutable, best-effort maps of Python module imports.

**This is static best-effort analysis, not a full type-checker.** Dynamic
imports, ``getattr``, and metaprogramming can hide edges; treat gaps as
"unknown", not "absent".  In particular, this package deliberately maps
module-level imports and top-level definitions only.  It does not claim to
know which functions or methods call one another: resolving that accurately
requires type inference and would create misleading confidence in v1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleNode:
    """One Python module (file) in the mapped codebase."""

    path: str
    module_name: str
    imports: tuple[str, ...]
    defines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "module_name": self.module_name,
                "imports": list(self.imports), "defines": list(self.defines)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModuleNode:
        path, name, imports, defines = (data.get(key) for key in ("path", "module_name", "imports", "defines"))
        if not isinstance(path, str) or not path or not isinstance(name, str) or not name:
            raise ValueError("module path and module_name must be non-empty strings")
        if not isinstance(imports, list) or not all(isinstance(item, str) for item in imports):
            raise ValueError("module imports must be a list of strings")
        if not isinstance(defines, list) or not all(isinstance(item, str) for item in defines):
            raise ValueError("module defines must be a list of strings")
        return cls(path, name, tuple(imports), tuple(defines))


@dataclass(frozen=True)
class CodebaseMap:
    """Full dependency graph for one scanned root, including parse gaps."""

    root_path: str
    modules: tuple[ModuleNode, ...]
    skipped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"root_path": self.root_path, "modules": [node.to_dict() for node in self.modules], "skipped": list(self.skipped)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodebaseMap:
        root, modules, skipped = data.get("root_path"), data.get("modules"), data.get("skipped", [])
        if not isinstance(root, str) or not root:
            raise ValueError("codebase map root_path must be a non-empty string")
        if not isinstance(modules, list) or not all(isinstance(item, dict) for item in modules):
            raise ValueError("codebase map modules must be a list of dicts")
        if not isinstance(skipped, list) or not all(isinstance(item, str) for item in skipped):
            raise ValueError("codebase map skipped must be a list of strings")
        return cls(root, tuple(ModuleNode.from_dict(item) for item in modules), tuple(skipped))

    def _by_name(self) -> dict[str, ModuleNode]:
        return {node.module_name: node for node in self.modules}

    def importers_of(self, module_name: str) -> tuple[str, ...]:
        """Return modules that directly import *module_name*."""
        return tuple(sorted(node.module_name for node in self.modules if module_name in node.imports))

    def imports_of(self, module_name: str) -> tuple[str, ...]:
        """Return the direct imports of *module_name*, or an empty tuple."""
        node = self._by_name().get(module_name)
        return node.imports if node is not None else ()

    def transitive_importers_of(self, module_name: str, *, max_depth: int | None = None) -> tuple[str, ...]:
        """Breadth-first reverse-edge traversal, ordered by distance then name."""
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        seen = {module_name}
        queue: deque[tuple[str, int]] = deque([(module_name, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for importer in self.importers_of(current):
                if importer not in seen:
                    seen.add(importer)
                    result.append(importer)
                    queue.append((importer, depth + 1))
        return tuple(result)
