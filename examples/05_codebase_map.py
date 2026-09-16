"""Offline demo: query a cached map of a small temporary Python project."""
from __future__ import annotations

import tempfile
from pathlib import Path

from kinetic_sdk.codemap import CodebaseMapTool

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "data.py").write_text("class Store: pass\n")
    (root / "app" / "service.py").write_text("from . import data\ndef run(): pass\n")
    (root / "app" / "cli.py").write_text("from app import service\n")
    tool = CodebaseMapTool(str(root))
    for query, module in (("overview", None), ("imports_of", "app.service"), ("importers_of", "app.data"), ("impact_of", "app.data")):
        result = tool.execute(query=query, module=module)
        print(result.error or result.output)
