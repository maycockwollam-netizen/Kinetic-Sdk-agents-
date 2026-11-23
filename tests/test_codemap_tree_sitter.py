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


def _writer(root):
    def write(path: str, content: str) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return write
