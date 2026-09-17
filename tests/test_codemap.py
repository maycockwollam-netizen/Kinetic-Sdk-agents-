"""Real-file AST tests for the static codebase map."""
from __future__ import annotations

import json

from kinetic_sdk.codemap import (
    CodebaseMap,
    CodebaseMapCache,
    CodebaseMapTool,
    build_codebase_map,
)


def write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_build_resolves_imports_and_collects_module_scope_defines(tmp_path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/sibling.py", "class Sibling: pass\n")
    write(tmp_path, "pkg/main.py", "import os.path\nfrom pkg import sibling as local\nfrom . import sibling\ndef direct(): pass\nif True:\n class Conditional: pass\n")
    result = build_codebase_map(tmp_path)
    nodes = {node.module_name: node for node in result.modules}
    assert nodes["pkg.main"].imports == ("os.path", "pkg.sibling")
    assert nodes["pkg.main"].defines == ("Conditional", "direct")


def test_from_import_distinguishes_submodules_attributes_and_star_imports(tmp_path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/sub.py", "class SomeClass: pass\n")
    write(tmp_path, "pkg/using_submodule.py", "from pkg import sub\n")
    write(tmp_path, "pkg/using_attribute.py", "from pkg.sub import SomeClass\n")
    write(tmp_path, "pkg/using_star.py", "from pkg.sub import *\n")

    result = build_codebase_map(tmp_path)
    nodes = {node.module_name: node for node in result.modules}

    assert nodes["pkg.using_submodule"].imports == ("pkg.sub",)
    assert nodes["pkg.using_attribute"].imports == ("pkg.sub",)
    assert nodes["pkg.using_star"].imports == ("pkg.sub",)
    assert result.importers_of("pkg.sub") == (
        "pkg.using_attribute",
        "pkg.using_star",
        "pkg.using_submodule",
    )


def test_package_root_normalizes_absolute_imports_to_map_module_names(tmp_path):
    package = tmp_path / "app"
    write(package, "__init__.py", "")
    write(package, "core.py", "class Engine: pass\n")
    write(package, "consumer.py", "from app.core import Engine\n")

    result = build_codebase_map(package)

    assert result.imports_of("consumer") == ("core",)
    assert result.importers_of("core") == ("consumer",)


def test_syntax_error_is_visible_but_other_modules_remain_mapped(tmp_path):
    write(tmp_path, "good.py", "def fine(): pass\n")
    write(tmp_path, "broken.py", "def nope(:\n")
    result = build_codebase_map(tmp_path)
    nodes = {node.module_name: node for node in result.modules}
    assert result.skipped == ("broken.py",)
    assert nodes["broken"].imports == nodes["broken"].defines == ()
    assert nodes["good"].defines == ("fine",)


def test_direct_and_transitive_reverse_edges_are_breadth_first(tmp_path):
    write(tmp_path, "a.py", "import b\n")
    write(tmp_path, "b.py", "import c\n")
    write(tmp_path, "c.py", "import d\n")
    write(tmp_path, "d.py", "")
    result = build_codebase_map(tmp_path)
    assert result.imports_of("c") == ("d",)
    assert result.importers_of("d") == ("c",)
    assert result.transitive_importers_of("d") == ("c", "b", "a")
    assert result.transitive_importers_of("d", max_depth=2) == ("c", "b")


def test_excluded_directories_are_not_walked(tmp_path):
    write(tmp_path, "kept.py", "")
    write(tmp_path, ".venv/hidden.py", "def hidden(): pass\n")
    write(tmp_path, "__pycache__/also_hidden.py", "")
    assert [node.path for node in build_codebase_map(tmp_path).modules] == ["kept.py"]


def test_cache_round_trips_and_invalidates_changed_files(tmp_path):
    write(tmp_path, "mod.py", "def old(): pass\n")
    cache = CodebaseMapCache(str(tmp_path))
    first = cache.get_or_build()
    second = cache.get_or_build()
    assert first == second
    payload = json.loads(cache.path.read_text(encoding="utf-8"))
    assert CodebaseMap.from_dict(payload["codebase_map"]) == first
    write(tmp_path, "mod.py", "def changed_to_a_longer_name(): pass\n")
    rebuilt = cache.get_or_build()
    assert rebuilt != first
    assert rebuilt.modules[0].defines == ("changed_to_a_longer_name",)


def test_tool_renders_all_queries_and_suggests_typo(tmp_path):
    write(tmp_path, "a.py", "import b\n")
    write(tmp_path, "b.py", "import c\n")
    write(tmp_path, "c.py", "")
    tool = CodebaseMapTool(str(tmp_path))
    assert "3 modules" in tool.execute(query="overview").output
    assert "b" in tool.execute(query="imports_of", module="a").output
    assert "a" in tool.execute(query="importers_of", module="b").output
    impact = tool.execute(query="impact_of", module="c", max_depth=2).output
    assert "hop 1: b" in impact and "hop 2: a" in impact
    for query in ("imports_of", "importers_of", "impact_of"):
        error = tool.execute(query=query, module="bee")
        assert error.is_error and "Suggestions: b." in error.error
