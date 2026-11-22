from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from kinetic_sdk.browser import BrowserTool
from kinetic_sdk.workspace.base import CommandResult, WorkspaceBase

# --- fake Playwright objects (no real Chromium needed) ----------------------

class FakePage:
    def __init__(self, url="about:blank", title="Tab"):
        self.url = url
        self._closed = False
        self._title = title
        self._handlers = {}

    def is_closed(self):
        return self._closed

    def close(self):
        self._closed = True

    def title(self):
        return self._title

    def goto(self, url):
        self.url = url

    def click(self, selector, timeout=None):
        pass

    def fill(self, selector, text, timeout=None):
        pass

    def screenshot(self, type="png"):
        return b"PNG-BYTES"

    def wait_for_selector(self, selector, timeout=None):
        pass

    def on(self, event, handler):
        self._handlers[event] = handler


class FakeBrowser:
    def __init__(self, connected=True):
        self._connected = connected
        self.pages = []
        self.closed = False

    def is_connected(self):
        return self._connected

    def close(self):
        self.closed = True

    def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page


class FakeChromium:
    def __init__(self, factory):
        self._factory = factory

    def launch(self, headless=True):
        return self._factory()

    def connect_over_cdp(self, url):
        raise RuntimeError("no cdp endpoint")


class FakePlaywright:
    def __init__(self, factory):
        self.chromium = FakeChromium(factory)
        self.stopped = False

    def stop(self):
        self.stopped = True


class DummyWorkspace(WorkspaceBase):
    """Non-Docker workspace used to verify the local-launch fallback."""

    def __init__(self):
        self._root = "/workspace"

    @property
    def root_path(self) -> str:
        return self._root

    def run_command(self, command, *, timeout=None, cwd=None) -> CommandResult:
        return CommandResult(output="", exit_code=0, duration_seconds=0.0)

    def read_text(self, relative_path: str) -> str:
        return ""

    def write_text(self, relative_path: str, content: str) -> None:
        return None

    def delete_file(self, relative_path: str) -> None:
        return None

    def list_files(self, pattern: str | None = None) -> list[str]:
        return []

    def list_directory(self, relative_path: str = ".") -> list[str]:
        return []


def _tool_with_fake_browser(factory, *, connected=True, workspace=None):
    tool = BrowserTool(workspace=workspace)
    tool._playwright = FakePlaywright(factory)
    tool._browser = FakeBrowser(connected=connected)
    return tool


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


def test_multitab_new_switch_list_close():
    tool = _tool_with_fake_browser(FakeBrowser)

    assert not tool.execute(action="new_tab").is_error
    assert not tool.execute(action="new_tab").is_error
    tabs = tool.execute(action="list_tabs").output
    assert len(tabs) == 2
    assert tabs[1]["active"] is True

    assert not tool.execute(action="switch_tab", index=0).is_error
    tabs = tool.execute(action="list_tabs").output
    assert tabs[0]["active"] is True

    assert not tool.execute(action="close_tab").is_error
    tabs = tool.execute(action="list_tabs").output
    assert len(tabs) == 1
    assert tool.execute(action="switch_tab", index=5).is_error


def test_recovery_when_active_page_closed():
    first = FakePage()
    first._closed = True  # simulate the active page crashing

    tool = BrowserTool()
    tool._playwright = FakePlaywright(FakeBrowser)
    tool._browser = FakeBrowser()
    tool._pages = [first]
    tool._active_index = 0

    result = tool.execute(action="navigate", url="http://example.com")
    assert not result.is_error
    # a fresh page replaced the dead one
    assert len(tool._pages) == 1
    assert tool._pages[0] is not first
    assert tool._pages[0].url == "http://example.com"


def test_recovery_when_browser_disconnected():
    def factory():
        return FakeBrowser(connected=True)

    tool = _tool_with_fake_browser(factory, connected=False)  # dead browser
    result = tool.execute(action="navigate", url="http://example.com")
    assert not result.is_error
    assert tool._browser.is_connected()
    assert tool.launch_mode == "local"


def test_fallback_local_launch_for_non_docker_workspace():
    tool = _tool_with_fake_browser(FakeBrowser, workspace=DummyWorkspace())
    result = tool.execute(action="navigate", url="http://example.com")
    assert not result.is_error
    assert tool.launch_mode == "local"


def test_close_is_idempotent_and_clears_pages():
    tool = _tool_with_fake_browser(FakeBrowser)
    tool.execute(action="new_tab")
    tool.close()
    assert tool._pages == []
    assert tool._browser is None
    assert tool._playwright is None
    tool.close()  # second close must not raise
