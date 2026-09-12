from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from kinetic_sdk.browser import BrowserTool


def test_constructor_is_lazy_when_playwright_is_unavailable(monkeypatch):
    tool = BrowserTool()
    real_import = builtins.__import__

    def no_playwright(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("missing playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_playwright)
    with pytest.raises(ImportError, match="kinetic-agent-sdk\\[browser\\]"):
        tool.execute(action="screenshot")


def test_static_page_navigation_console_and_screenshot(tmp_path: Path):
    pytest.importorskip("playwright.sync_api")
    page = tmp_path / "page.html"
    page.write_text("<script>console.log('ready')</script><h1>Hello</h1>", encoding="utf-8")
    with BrowserTool() as tool:
        assert not tool.execute(action="navigate", url=page.as_uri()).is_error
        logs = tool.execute(action="get_console_logs")
        assert any(item["text"] == "ready" for item in logs.output)
        shot = tool.execute(action="screenshot")
        assert shot.output == "screenshot captured"
        assert shot.metadata["image_base64"]
