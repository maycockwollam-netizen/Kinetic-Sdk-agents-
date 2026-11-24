"""AST-based construction of the codebase's deliberately shallow import map."""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterable
from pathlib import Path

from kinetic_sdk.codemap.models import CodebaseMap, ModuleNode
from kinetic_sdk.codemap.tree_sitter import TreeSitterScanner
from kinetic_sdk.workspace.manager import LocalWorkspace

#: Shared optional tree-sitter scanner (lazy — no-op until first use).
_TREE_SITTER = TreeSitterScanner()

DEFAULT_EXCLUDE = (".venv", "venv", "node_modules", "__pycache__", ".git")

#: File extensions each supported language is discovered under. Python is
#: parsed with the stdlib ``ast`` (exact); the others are best-effort
#: structural scans with no third-party tree-sitter dependency.
LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".mjs", ".cjs"),
    "typescript": (".ts", ".mts", ".cts", ".tsx"),
    "rust": (".rs",),
    "go": (".go",),
}

#: Languages enabled by default. Pass ``languages=("python", "javascript", ...)``
#: to scan more; a file whose extension matches several enabled languages is
#: assigned to the FIRST matching language in this order.
DEFAULT_LANGUAGES: tuple[str, ...] = ("python",)


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


def build_codebase_map(
    root_path: str | os.PathLike[str],
    *,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    use_tree_sitter: bool = False,
) -> CodebaseMap:
    """Walk *root_path* and build a static import map for the enabled languages.

    ``python`` is parsed exactly with the stdlib ``ast`` (relative imports,
    attribute-vs-submodule resolution). The other supported languages
    (``javascript`` / ``typescript`` / ``rust`` / ``go``) are best-effort
    structural scans with no extra dependency: imports/requires/uses are
    captured as raw spec strings and top-level declarations (functions,
    classes, structs, enums, consts, ...) as name strings. Parse and decoding
    failures stay visible as empty nodes in ``skipped``.

    Args:
        root_path: Repository root to walk.
        exclude: Directory names skipped when walking.
        languages: Enabled language names (subset of
            :data:`LANGUAGE_EXTENSIONS` keys). A file whose extension belongs
            to several enabled languages is assigned to the FIRST in this order.
        use_tree_sitter: When ``True`` (and the optional ``[codemap]`` extra is
            installed), non-Python languages are parsed exactly with
            tree-sitter; otherwise they keep the best-effort regex scan.
    """
    workspace = LocalWorkspace(root_path)
    root = Path(workspace.root_path)
    excluded = set(exclude)
    enabled = [lang for lang in languages if lang in LANGUAGE_EXTENSIONS]
    ext_to_lang: dict[str, str] = {}
    for lang in enabled:
        for ext in LANGUAGE_EXTENSIONS[lang]:
            ext_to_lang.setdefault(ext, lang)

    grouped: dict[str, list[tuple[Path, str, str]]] = {lang: [] for lang in enabled}
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        for filename in sorted(filenames):
            file_lang = ext_to_lang.get(Path(filename).suffix)
            if file_lang is None:
                continue
            absolute = Path(directory) / filename
            relative = absolute.relative_to(root).as_posix()
            grouped[file_lang].append((absolute, relative, _module_name(relative)))

    nodes: list[ModuleNode] = []
    skipped: list[str] = []
    for lang, files in grouped.items():
        if lang == "python":
            nodes.extend(_build_python_nodes(root, files, skipped))
        else:
            for absolute, relative, module_name in files:
                try:
                    text = absolute.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    nodes.append(ModuleNode(relative, module_name, (), ()))
                    skipped.append(relative)
                    continue
                # The file suffix picks the grammar variant (e.g. ".tsx" needs
                # the TSX grammar, not plain TypeScript).
                scanned = _TREE_SITTER.scan(lang, text, variant=absolute.suffix) if use_tree_sitter else None
                if scanned is not None:
                    imports, defines = scanned
                else:
                    imports = _structural_imports(lang, text)
                    defines = _structural_defines(lang, text)
                nodes.append(
                    ModuleNode(
                        relative,
                        module_name,
                        tuple(sorted(imports)),
                        tuple(sorted(defines)),
                    )
                )
    return CodebaseMap(str(root), tuple(sorted(nodes, key=lambda item: item.module_name)), tuple(sorted(skipped)))


def _build_python_nodes(
    root: Path,
    files: list[tuple[Path, str, str]],
    skipped: list[str],
) -> list[ModuleNode]:
    """Parse the discovered ``.py`` files with the stdlib ``ast``."""
    known_modules = {module_name for _, _, module_name in files}
    package_name = root.name if (root / "__init__.py").is_file() else None
    nodes: list[ModuleNode] = []
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
    return nodes


# --- best-effort structural scanners for the non-Python languages ----------

_JS_IMPORT_PATTERNS = (
    re.compile(r"""\bfrom\s+['"]([^'"]+)['"]"""),
    re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\s+['"]([^'"]+)['"]"""),
)
_JS_DEFINE_PATTERNS = (
    re.compile(r"""\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"""),
    re.compile(r"""\b(?:export\s+)?class\s+([A-Za-z_$][\w$]*)"""),
    re.compile(r"""\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="""),
)
_RUST_IMPORT_PATTERNS = (
    re.compile(r"\buse\s+([^;\n]+);"),
    re.compile(r"\bextern\s+crate\s+([a-z_0-9:]+)"),
)
_RUST_DEFINE_PATTERNS = (
    re.compile(r"\bfn\s+([a-z_][a-z0-9_]*)"),
    re.compile(r"\bstruct\s+([A-Z][A-Za-z0-9_]*)"),
    re.compile(r"\benum\s+([A-Z][A-Za-z0-9_]*)"),
    re.compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_:]*)\s*="),
    re.compile(r"\bconst\s+([A-Z_][A-Z0-9_]*)\s*:"),
)
_GO_IMPORT_PATTERNS = (
    re.compile(r'import\s+[a-zA-Z_]+\s+"([^"]+)"'),
    re.compile(r'import\s+"([^"]+)"'),
    re.compile(r'import\s*\(([^)]*)\)'),
)
_GO_DEFINE_PATTERNS = (
    re.compile(r"\bfunc\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)\s*\("),
    re.compile(r"\btype\s+([A-Za-z_]\w*)\s+(struct|interface)\b"),
    re.compile(r"\btype\s+([A-Za-z_]\w*)\s*="),
    re.compile(r"\b(?:const|var)\s+([A-Za-z_]\w*)\s*="),
)


def _structural_imports(language: str, text: str) -> list[str]:
    """Best-effort list of import spec strings for a non-Python language."""
    if language in ("javascript", "typescript"):
        js_specs: list[str] = []
        for pattern in _JS_IMPORT_PATTERNS:
            js_specs.extend(m.group(1).strip() for m in pattern.finditer(text))
        return [spec for spec in js_specs if spec]
    if language == "rust":
        rust_specs: list[str] = []
        for match in _RUST_IMPORT_PATTERNS[0].finditer(text):
            rust_specs.append(match.group(1).strip().rstrip(";").strip())
        for match in _RUST_IMPORT_PATTERNS[1].finditer(text):
            rust_specs.append(match.group(1).strip())
        return [spec for spec in rust_specs if spec]
    if language == "go":
        go_specs: list[str] = []
        for match in _GO_IMPORT_PATTERNS[0].finditer(text):
            go_specs.append(match.group(1))
        for match in _GO_IMPORT_PATTERNS[1].finditer(text):
            go_specs.append(match.group(1))
        for block in _GO_IMPORT_PATTERNS[2].finditer(text):
            go_specs.extend(re.findall(r'"([^"]+)"', block.group(1)))
        return [spec for spec in go_specs if spec]
    return []


def _structural_defines(language: str, text: str) -> list[str]:
    """Best-effort top-level declaration names for a non-Python language."""
    patterns: tuple[re.Pattern[str], ...]
    if language in ("javascript", "typescript"):
        patterns = _JS_DEFINE_PATTERNS
    elif language == "rust":
        patterns = _RUST_DEFINE_PATTERNS
    elif language == "go":
        patterns = _GO_DEFINE_PATTERNS
    else:
        return []
    found: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            if name:
                found.add(name)
    return sorted(found)
