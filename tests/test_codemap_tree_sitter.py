"""Optional tree-sitter exact scanning for the codebase map."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter")

from kinetic_sdk.codemap import TreeSitterScanner, build_codebase_map  # noqa: E402


def _scan(language: str, text: str):
    scanner = TreeSitterScanner()
    result = scanner.scan(language, text)
    assert result is not None, f"grammar {language!r} should be available"
    return result


def test_tree_sitter_javascript_imports_and_defines() -> None:
    imports, defines = _scan(
        "javascript",
        'import {x} from "react";\nimport "./style.css";\nexport function render() {}\nconst N = require("mod");\nexport class Widget {}\n',
    )
    assert set(imports) == {"./style.css", "react", "mod"}
    assert {"render", "N", "Widget"} <= set(defines)


def test_tree_sitter_typescript() -> None:
    imports, defines = _scan(
        "typescript", 'import express from "express";\nexport async function serve() {}\n'
    )
    assert imports == ["express"]
    assert defines == ["serve"]


def test_tree_sitter_rust() -> None:
    imports, defines = _scan(
        "rust",
        "use std::collections::HashMap;\nextern crate serde;\n"
        "fn main() {}\nstruct Config {}\nenum Mode {}\nconst MAX: u32 = 1;\n",
    )
    assert set(imports) == {"std::collections::HashMap", "serde"}
    assert {"main", "Config", "Mode", "MAX"} <= set(defines)


def test_tree_sitter_go() -> None:
    imports, defines = _scan(
        "go",
        'import (\n  "fmt"\n  "os"\n)\nimport "math"\n'
        "func main() {}\ntype Config struct {}\nconst Version = \"1\"\nvar x = 2\n",
    )
    assert set(imports) == {"fmt", "os", "math"}
    assert {"main", "Config", "Version", "x"} <= set(defines)


def test_builder_use_tree_sitter_js(tmp_path) -> None:
    write = _writer(tmp_path)
    write("app.js", 'import {x} from "react";\nimport "./style.css";\nexport function render() {}\nexport class Widget {}\n')
    cm = build_codebase_map(tmp_path, languages=("javascript",), use_tree_sitter=True)
    node = next(n for n in cm.modules if n.module_name == "app")
    assert set(node.imports) == {"./style.css", "react"}
    assert {"render", "Widget"} <= set(node.defines)


def test_tree_sitter_captures_dynamic_imports() -> None:
    # A dynamic import("...") inside a function body is still an import spec;
    # `import` as a dynamic-call callee is the ``import`` token node, not an
    # identifier, which previously slipped through.
    imports, _ = _scan(
        "javascript",
        'export async function load() { const m = await import("./mod.js"); return m; }\n'
        'function legacy() { const x = require("old"); return x; }\n',
    )
    assert set(imports) == {"./mod.js", "old"}


def test_tree_sitter_only_extracts_top_level_definitions() -> None:
    imports, defines = _scan(
        "javascript",
        "function outer() {\n  function inner() {}\n  const local = 123;\n}\n"
        "export function render() {}\nclass Widget { method() {} }\n",
    )
    assert {"outer", "render", "Widget"} <= set(defines)
    assert "inner" not in defines
    assert "local" not in defines
    assert "method" not in defines

    _, rust_defines = _scan(
        "rust",
        "fn main() { fn helper() {} }\nstruct S {}\nmod m { fn nested() {} }\n",
    )
    assert {"main", "S"} <= set(rust_defines)
    assert "helper" not in rust_defines
    assert "nested" not in rust_defines


def test_tree_sitter_tsx_uses_the_tsx_grammar() -> None:
    jssx = "export const App = () => <div>{x}</div>;\n"
    # Plain TypeScript grammar cannot parse JSX -> nothing extracted.
    assert _scan("typescript", jssx) == ([], [])
    # The ".tsx" variant uses the TSX grammar and extracts the declaration.
    scanner = TreeSitterScanner()
    tsx = scanner.scan("typescript", jssx, variant=".tsx")
    assert tsx is not None
    imports, defines = tsx
    assert imports == []
    assert defines == ["App"]


def test_builder_use_tree_sitter_tsx(tmp_path) -> None:
    write = _writer(tmp_path)
    write("app.tsx", "export const App = () => <div>{x}</div>;\n")
    cm = build_codebase_map(tmp_path, languages=("typescript",), use_tree_sitter=True)
    node = next(n for n in cm.modules if n.module_name == "app")
    assert node.defines == ("App",)


def _writer(root):
    def write(path: str, content: str) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return write
