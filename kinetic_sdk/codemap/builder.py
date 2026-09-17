"""AST-based construction of the codebase's deliberately shallow import map."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable
from pathlib import Path

from kinetic_sdk.codemap.models import CodebaseMap, ModuleNode
from kinetic_sdk.workspace.manager import LocalWorkspace

DEFAULT_EXCLUDE = (".venv", "venv", "node_modules", "__pycache__", ".git")


def _module_name(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "__init__"


def _relative_base(module_name: str, level: int, module: str | None, is_package: bool) -> str:
    parts = module_name.split(".")
    # A package __init__ has its own package as its import context; normal
    # modules import relative to their containing package.
    package = parts if is_package else parts[:-1]
    up = level - 1
    if up > len(package):
        prefix = "." * level
        return prefix + (module or "")
    base = package[: len(package) - up]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _canonical_module_name(name: str, known_modules: set[str], package_name: str | None) -> str:
    """Return the map's module name for an import rooted at this package."""
    if package_name and name.startswith(f"{package_name}."):
        relative_name = name.removeprefix(f"{package_name}.")
        if relative_name in known_modules:
            return relative_name
    return name


def _imports(
    tree: ast.Module,
    module_name: str,
    is_package: bool,
    known_modules: set[str],
    package_name: str | None,
) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(_canonical_module_name(alias.name, known_modules, package_name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = _relative_base(module_name, node.level, node.module, is_package)
            base = _canonical_module_name(base, known_modules, package_name)
            if not base:
                imports.update(alias.name for alias in node.names)
                continue
            for alias in node.names:
                if alias.name == "*":
                    imports.add(base)
                    continue
                candidate = f"{base}.{alias.name}"
                # ``from package import child`` imports a child module, while
                # ``from module import Name`` imports an attribute defined by
                # its base module. AST alone cannot distinguish them, so use
                # the modules discovered for this codebase.
                imports.add(candidate if candidate in known_modules else base)
    return imports


def _defines(tree: ast.Module) -> set[str]:
    """Collect definitions in module-level statement blocks, never nested bodies."""
    found: set[str] = set()

    def visit_statements(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(statement.name)
            elif isinstance(statement, ast.If):
                visit_statements(statement.body)
                visit_statements(statement.orelse)
            elif isinstance(statement, ast.Try) or (hasattr(ast, "TryStar") and isinstance(statement, ast.TryStar)):
                visit_statements(statement.body)
                visit_statements(statement.orelse)
                visit_statements(statement.finalbody)
                for handler in statement.handlers:
                    visit_statements(handler.body)

    visit_statements(tree.body)
    return found


def build_codebase_map(root_path: str | os.PathLike[str], *, exclude: Iterable[str] = DEFAULT_EXCLUDE) -> CodebaseMap:
    """Walk *root_path*, parse every Python file, and build a static map.

    Parse and decoding failures remain visible as empty nodes in ``skipped``;
    a partial graph is useful only when its blind spots are explicit.
    """
    workspace = LocalWorkspace(root_path)
    root = Path(workspace.root_path)
    excluded = set(exclude)
    files: list[tuple[Path, str, str]] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                absolute = Path(directory) / filename
                relative = absolute.relative_to(root).as_posix()
                files.append((absolute, relative, _module_name(relative)))
    known_modules = {module_name for _, _, module_name in files}
    package_name = root.name if (root / "__init__.py").is_file() else None
    nodes: list[ModuleNode] = []
    skipped: list[str] = []
    for absolute, relative, module_name in files:
        is_package = absolute.name == "__init__.py"
        try:
            tree = ast.parse(absolute.read_text(encoding="utf-8"), filename=relative)
        except (SyntaxError, UnicodeDecodeError, OSError):
            nodes.append(ModuleNode(relative, module_name, (), ()))
            skipped.append(relative)
            continue
        nodes.append(
            ModuleNode(
                relative,
                module_name,
                tuple(sorted(_imports(tree, module_name, is_package, known_modules, package_name))),
                tuple(sorted(_defines(tree))),
            )
        )
    return CodebaseMap(str(root), tuple(sorted(nodes, key=lambda item: item.module_name)), tuple(sorted(skipped)))
