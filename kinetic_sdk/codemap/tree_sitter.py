"""Exact import/definition extraction for non-Python languages via tree-sitter.

The regex structural scan in :mod:`kinetic_sdk.codemap.builder` is best-effort;
when the optional ``[codemap]`` extra is installed, :class:`TreeSitterScanner`
parses javascript/typescript/rust/go grammatically and extracts import specs
and top-level declarations from the real syntax tree. The builder prefers this
scanner when ``use_tree_sitter=True`` and falls back to the regex scan when the
package (or one language grammar) is missing.

Everything is lazy: importing this module never imports ``tree_sitter``; the
first :meth:`TreeSitterScanner.scan` call loads the grammars it needs.
"""

from __future__ import annotations

import functools
import importlib

#: language name -> importable grammar module.
_GRAMMAR_MODULES = {
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
}

_IMPORT_NODES: dict[str, tuple[str, ...]] = {
    "javascript": ("import_statement", "call_expression"),
    "typescript": ("import_statement", "call_expression"),
    "rust": ("use_declaration", "extern_crate_declaration"),
    "go": ("import_declaration",),
}


@functools.lru_cache(maxsize=None)
def loaded_language(language: str, variant: str | None = None) -> object | None:
    """The tree-sitter ``Language`` for *language*, or ``None`` if unavailable.

    ``variant`` refines the grammar within one language family, currently only
    ``".tsx"`` for ``typescript`` (JSX is not valid plain TypeScript, so the
    TSX grammar must be picked per file).
    """
    module = _GRAMMAR_MODULES.get(language)
    if module is None:
        return None
    try:
        imported = importlib.import_module(module)
        from tree_sitter import Language

        if variant == ".tsx" and module == "tree_sitter_typescript":
            getter = getattr(imported, "language_tsx", None)
        else:
            # Most grammars expose ``language()``; tree-sitter-typescript
            # exposes ``language_typescript()`` / ``language_tsx()``.
            getter = getattr(imported, "language", None)
            if getter is None:
                getter = getattr(imported, "language_typescript", None)
        if getter is None:
            return None
        return Language(getter())
    except Exception:  # noqa: BLE001 - optional dependency may be missing
        return None


def _children(node: object, node_type: str) -> list:
    return [child for child in node.children if child.type == node_type]  # type: ignore[attr-defined]


def _text(node: object) -> str:
    return node.text.decode("utf-8", "replace")  # type: ignore[attr-defined]


def _first_child(node: object, node_type: str) -> str:
    for child in _children(node, node_type):
        return _text(child)
    return ""


def _descendant(node: object, node_type: str) -> object | None:
    if node.type == node_type:  # type: ignore[attr-defined]
        return node
    for child in node.children:  # type: ignore[attr-defined]
        found = _descendant(child, node_type)
        if found is not None:
            return found
    return None


def _import_specs(language: str, node: object) -> list[str]:
    node_type = node.type  # type: ignore[attr-defined]
    if language in ("javascript", "typescript"):
        if node_type == "import_statement":
            string = _descendant(node, "string")
            return [_strip_quotes(_text(string))] if string else []
        # require("x") / import("x") dynamic call
        call = _children(node, "identifier")
        if call and _text(call[0]) in ("require", "import"):
            string = _descendant(node, "string")
            return [_strip_quotes(_text(string))] if string else []
        return []
    if language == "rust":
        text = node.text.decode("utf-8", "replace")  # type: ignore[attr-defined]
        if node_type == "use_declaration":
            return [text[len("use "):].lstrip().rstrip(";").strip()]
        if node_type == "extern_crate_declaration":
            return [text[len("extern crate "):].lstrip().rstrip(";").strip()]
        return []
    if language == "go":
        specs: list[str] = []
        for literal in _descendants(node, "interpreted_string_literal"):
            specs.append(_strip_quotes(literal.text.decode("utf-8", "replace")))  # type: ignore[attr-defined]
        return [spec for spec in specs if spec]
    return []


def _descendants(node: object, node_type: str) -> list:
    found: list = []
    if node.type == node_type:  # type: ignore[attr-defined]
        found.append(node)
    for child in node.children:  # type: ignore[attr-defined]
        found.extend(_descendants(child, node_type))
    return found


def _define_names(language: str, node: object) -> list[str]:
    node_type = node.type  # type: ignore[attr-defined]
    if language in ("javascript", "typescript"):
        if node_type in ("function_declaration", "generator_function_declaration", "class_declaration"):
            name = _first_child(node, "identifier")
            return [name] if name else []
        if node_type == "lexical_declaration":  # const/let/var
            names: list[str] = []
            for declarator in _children(node, "variable_declarator"):
                name = _first_child(declarator, "identifier")
                if name:
                    names.append(name)
            return names
        return []
    if language == "rust":
        if node_type == "function_item":
            name = _first_child(node, "identifier")
        elif node_type in ("struct_item", "enum_item", "type_item"):
            name = _first_child(node, "type_identifier")
        elif node_type in ("const_item", "static_item"):
            name = _first_child(node, "identifier")
        else:
            return []
        return [name] if name else []
    if language == "go":
        if node_type == "function_declaration":
            name = _first_child(node, "identifier")
            return [name] if name else []
        if node_type == "type_declaration":
            return [
                name
                for spec in _children(node, "type_spec")
                if (name := _first_child(spec, "type_identifier"))
            ]
        if node_type in ("const_declaration", "var_declaration"):
            spec_type = "const_spec" if node_type == "const_declaration" else "var_spec"
            return [
                name
                for spec in _children(node, spec_type)
                if (name := _first_child(spec, "identifier"))
            ]
    return []


def _strip_quotes(text: str) -> str:
    return text.strip("\"'`").strip()


#: Node type of each language's root, whose DIRECT children are top-level.
_ROOT_TYPES: dict[str, tuple[str, ...]] = {
    "javascript": ("program",),
    "typescript": ("program",),
    "rust": ("source_file",),
    "go": ("source_file",),
}


class TreeSitterScanner:
    """Parse one non-Python source with tree-sitter and extract imports/defines.

    ``scan(language, text)`` returns ``(imports, defines)`` as sorted lists,
    or ``None`` when the grammar for *language* is not installed. Imports are
    collected anywhere in the file, but only TOP-LEVEL declarations become
    defines (nested functions/classes/consts are deliberately excluded).

    ``variant`` refines the grammar per file (currently ``".tsx"`` for
    ``typescript``) and is passed through to :func:`loaded_language`.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, object] = {}

    def available(self, language: str, variant: str | None = None) -> bool:
        return language in _GRAMMAR_MODULES and loaded_language(language, variant) is not None

    def scan(
        self, language: str, text: str, variant: str | None = None
    ) -> tuple[list[str], list[str]] | None:
        parser = self._parser(language, variant)
        if parser is None:
            return None

        tree = parser.parse(text.encode("utf-8"))  # type: ignore[attr-defined]
        imports: set[str] = set()
        defines: set[str] = set()
        self._walk(tree.root_node, language, imports, defines, top_level=True)  # type: ignore[attr-defined]
        return sorted(imports), sorted(defines)

    def _parser(self, language: str, variant: str | None = None) -> object | None:
        key = f"{language}:{variant or ''}"
        if key in self._parsers:
            return self._parsers[key]
        loaded = loaded_language(language, variant)
        if loaded is None:
            return None
        from tree_sitter import Parser

        parser = Parser(loaded)  # type: ignore[arg-type]  # starts as a Language at runtime
        self._parsers[key] = parser
        return parser

    def _walk(
        self,
        node: object,
        language: str,
        imports: set[str],
        defines: set[str],
        *,
        top_level: bool,
    ) -> None:
        node_type = node.type  # type: ignore[attr-defined]
        if node_type in _IMPORT_NODES[language]:
            imports.update(_import_specs(language, node))
        if top_level:
            defines.update(_define_names(language, node))
        # Definitions stay top-level only under the file root and, for
        # javascript/typescript, under an ``export`` wrapper whose declaration
        # IS a top-level item; anything deeper (function/class bodies, blocks)
        # never collects definitions.
        keep_top_level = top_level and (
            node_type in _ROOT_TYPES[language]
            or (
                language in ("javascript", "typescript")
                and node_type == "export_statement"
            )
        )
        for child in node.children:  # type: ignore[attr-defined]
            self._walk(child, language, imports, defines, top_level=keep_top_level)  # type: ignore[attr-defined]