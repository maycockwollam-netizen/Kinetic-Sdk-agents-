"""A stateful, optional Playwright browser tool for agent workflows."""

from __future__ import annotations

import base64
import time
import weakref
from typing import Any

from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase

# CDP port used when launching the browser inside a ``DockerWorkspace``.
_CDP_PORT = 9222

#: Default cap on a ``snapshot`` action's ``body_text``.
_DEFAULT_SNAPSHOT_CHARS = 8_000


class BrowserTool(Tool):
    """Control headless Playwright pages through a compact action schema.

    Playwright is imported only when the first action is executed, keeping the
    core SDK dependency-free.

    Sandboxing: when constructed with a :class:`DockerWorkspace` in ``exec``
    mode, the tool *attempts* to launch the browser *inside that container* and
    connect to it over CDP, so page navigation happens inside the sandbox
    rather than on the agent host. If the container lacks a browser binary, is
    unreachable from the agent process, or the CDP handshake fails, it falls
    back to launching a local headless Chromium and records the effective mode
    in :attr:`launch_mode`. For a non-Docker workspace (or ``None``) the
    browser always runs locally.

    Multiple tabs are supported. Each tab is a separate Playwright page; the
    active tab is the one that receives ``navigate``/``click``/``type``/etc.
    A crashed or closed page/browser is detected and re-launched automatically
    on the next action.
    """

    name = "browser"
    description = "Navigate and interact with a headless web page using CSS selectors."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [
                "navigate", "click", "type", "screenshot", "snapshot",
                "get_console_logs", "get_network_requests", "wait_for_selector",
                "new_tab", "close_tab", "switch_tab", "list_tabs",
                "get_downloads",
            ]},
            "url": {"type": "string"},
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "index": {"type": "integer", "minimum": 0},
            "timeout": {"type": "number", "minimum": 0},
            "max_chars": {"type": "integer", "minimum": 0, "description": "Optional cap on a snapshot's body_text."},
        },
        "required": ["action"],
    }

    def __init__(self, workspace: WorkspaceBase | None = None) -> None:
        self.workspace = workspace
        self._playwright: Any = None
        self._browser: Any = None
        self._pages: list[Any] = []
        self._active_index = 0
        self._console_logs: list[dict[str, str]] = []
        self._network_requests: list[dict[str, Any]] = []
        #: Weak-map of in-flight request object -> start time. Keying by the object
        # (not ``id()``) means aborted requests never leak and a GC-recycled
        # ``id()`` can never be attributed to a different request.
        self._request_started: "weakref.WeakKeyDictionary[Any, float]" = weakref.WeakKeyDictionary()
        self._downloads: list[dict[str, str]] = []
        # "local" | "docker-cdp" — recorded after the first launch.
        self.launch_mode = "local"

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "BrowserTool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close all pages/browser and stop Playwright (safe to call twice)."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._pages = []
        self._active_index = 0
        self._browser = self._playwright = None

    # -- browser/page management --------------------------------------------

    def _start_playwright(self) -> Any:
        if self._playwright is not None:
            return self._playwright
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "BrowserTool requires Playwright. Install it with: "
                "pip install kinetic-agent-sdk[browser]"
            ) from exc
        self._playwright = sync_playwright().start()
        return self._playwright

    def _teardown_browser(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
        self._browser = None
        self._pages = []
        self._active_index = 0

    def _ensure_browser(self) -> Any:
        """Return a live browser, re-launching it if the current one died."""
        if self._browser is not None:
            try:
                if self._browser.is_connected():
                    return self._browser
            except Exception:
                pass
            self._teardown_browser()
        self._start_playwright()
        browser, mode = self._try_launch_via_workspace()
        if browser is None:
            browser = self._playwright.chromium.launch(headless=True)
            mode = "local"
        self._browser = browser
        self.launch_mode = mode
        return self._browser

    def _try_launch_via_workspace(self) -> tuple[Any, str]:
        """Launch the browser inside a ``DockerWorkspace`` (exec mode) via CDP.

        Returns ``(browser, mode)``. On any failure returns
        ``(None, "local")`` so the caller falls back to a local launch.
        """
        ws = self.workspace
        if ws is None:
            return None, "local"
        try:
            from kinetic_sdk.workspace.docker_workspace import DockerWorkspace
        except Exception:
            return None, "local"
        if not isinstance(ws, DockerWorkspace):
            return None, "local"
        container = ws.container_name
        if not container:
            return None, "local"
        launch_cmd = (
            "nohup chromium --headless=new --no-sandbox "
            f"--remote-debugging-port={_CDP_PORT} "
            "--user-data-dir=/tmp/kinetic-browser-cdp "
            ">/dev/null 2>&1 &"
        )
        result = ws.run_command(launch_cmd, timeout=20)
        if result.exit_code != 0:
            return None, "local"
        ip = self._container_ip(container)
        if not ip:
            return None, "local"
        try:
            browser = self._playwright.chromium.connect_over_cdp(
                f"http://{ip}:{_CDP_PORT}"
            )
        except Exception:
            return None, "local"
        return browser, "docker-cdp"

    @staticmethod
    def _container_ip(container: str) -> str:
        """Return the container's IP address, or ``""`` if it cannot be found."""
        import subprocess

        try:
            proc = subprocess.run(
                [
                    "docker", "inspect", "-f",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                    container,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return ""
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    def _ensure_active_page(self) -> Any:
        """Return the active page, creating/recovering it as needed."""
        browser = self._ensure_browser()
        if self._pages:
            page = self._pages[self._active_index]
            try:
                if not page.is_closed():
                    return page
            except Exception:
                pass
            # Active page died -> drop it and fall through to create a new one.
            self._pages.pop(self._active_index)
            if self._active_index >= len(self._pages):
                self._active_index = max(0, len(self._pages) - 1)
        page = browser.new_page()
        self._attach_handlers(page)
        self._pages.append(page)
        self._active_index = len(self._pages) - 1
        return page

    def _attach_handlers(self, page: Any) -> None:
        page.on("console", self._record_console)
        page.on("request", self._record_request_start)
        page.on("response", self._record_response)
        page.on("download", self._record_download)

    def _record_console(self, message: Any) -> None:
        self._console_logs.append({"type": message.type, "text": message.text})

    def _record_request_start(self, request: Any) -> None:
        self._request_started[request] = time.monotonic()

    def _record_response(self, response: Any) -> None:
        request = response.request
        started = self._request_started.pop(request, time.monotonic())
        self._network_requests.append(
            {"url": request.url, "method": request.method, "status": response.status,
             "duration": round(time.monotonic() - started, 4)}
        )

    def _record_download(self, download: Any) -> None:
        try:
            filename = download.suggested_filename
        except Exception:
            filename = ""
        self._downloads.append({"filename": filename})

    # -- action dispatch -----------------------------------------------------

    def execute(self, action: str, **params: Any) -> ToolResult:  # type: ignore[override]
        """Run one browser action, recovering once from a crash/close."""
        try:
            return self._execute_once(action, params)
        except ImportError:
            raise
        except Exception as exc:
            if self._recover():
                try:
                    return self._execute_once(action, params)
                except Exception as exc2:
                    return ToolResult(
                        error=f"browser {action} failed after recovery: "
                              f"{type(exc2).__name__}: {exc2}"
                    )
            return ToolResult(
                error=f"browser {action} failed: {type(exc).__name__}: {exc}"
            )

    def _recover(self) -> bool:
        self._teardown_browser()
        try:
            self._ensure_browser()
            return True
        except Exception:
            return False

    def _execute_once(self, action: str, params: dict[str, Any]) -> ToolResult:
        if action in ("new_tab", "close_tab", "switch_tab", "list_tabs", "get_downloads"):
            return self._tab_action(action, params)
        page = self._ensure_active_page()
        if action == "navigate":
            url = self._required_string(params, "url")
            page.goto(url)
            return ToolResult(output=f"navigated to {url}")
        if action == "click":
            page.click(self._required_string(params, "selector"), timeout=self._timeout(params))
            return ToolResult(output="click completed")
        if action == "type":
            page.fill(
                self._required_string(params, "selector"),
                self._required_string(params, "text"),
                timeout=self._timeout(params),
            )
            return ToolResult(output="text entered")
        if action == "screenshot":
            image = page.screenshot(type="png")
            return ToolResult(
                output="screenshot captured",
                metadata={"image_base64": base64.b64encode(image).decode("ascii"),
                          "media_type": "image/png"},
            )
        if action == "snapshot":
            return self._snapshot(page, params)
        if action == "get_console_logs":
            return ToolResult(output=list(self._console_logs))
        if action == "get_network_requests":
            return ToolResult(output=list(self._network_requests))
        if action == "wait_for_selector":
            page.wait_for_selector(
                self._required_string(params, "selector"), timeout=self._timeout(params)
            )
            return ToolResult(output="selector found")
        return ToolResult(error=f"unsupported browser action: {action!r}")

    def _snapshot(self, page: Any, params: dict[str, Any]) -> ToolResult:
        """A readable text snapshot of the active page (URL + title + body)."""
        url: Any = ""
        title: Any = ""
        body: Any = ""
        for label, getter in (
            ("title", lambda: page.title()),
            ("body", lambda: page.inner_text("body")),
        ):
            try:
                value = getter()
                if label == "title":
                    title = value or ""
                else:
                    body = value or ""
            except Exception:  # noqa: BLE001 - any page API may be unavailable
                pass
        try:
            url = page.url or ""
        except Exception:  # noqa: BLE001
            url = ""
        raw = params.get("max_chars")
        limit = _DEFAULT_SNAPSHOT_CHARS if raw is None else max(0, int(raw))
        body = body.strip()
        truncated = len(body) > limit
        if truncated:
            body = body[:limit] + "...(cut)"
        return ToolResult(
            output={
                "url": url,
                "title": title,
                "body_text": body,
                "truncated": truncated,
            }
        )

    def _tab_action(self, action: str, params: dict[str, Any]) -> ToolResult:
        if action == "list_tabs":
            tabs = []
            for i, page in enumerate(self._pages):
                try:
                    tabs.append({"index": i, "url": page.url, "title": page.title(),
                                 "active": i == self._active_index})
                except Exception:
                    tabs.append({"index": i, "url": "", "title": "",
                                 "active": i == self._active_index})
            return ToolResult(output=tabs)
        if action == "new_tab":
            browser = self._ensure_browser()
            page = browser.new_page()
            self._attach_handlers(page)
            self._pages.append(page)
            self._active_index = len(self._pages) - 1
            return ToolResult(output=f"opened tab {self._active_index}")
        if action == "close_tab":
            if not self._pages:
                return ToolResult(error="no open tabs")
            page = self._pages.pop(self._active_index)
            try:
                page.close()
            except Exception:
                pass
            if self._active_index >= len(self._pages):
                self._active_index = max(0, len(self._pages) - 1)
            return ToolResult(output="tab closed")
        if action == "switch_tab":
            index = params.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(self._pages):
                return ToolResult(error=f"tab index out of range: {index!r}")
            self._active_index = index
            return ToolResult(output=f"switched to tab {index}")
        if action == "get_downloads":
            return ToolResult(output=list(self._downloads))
        return ToolResult(error=f"unsupported browser action: {action!r}")

    @staticmethod
    def _required_string(params: dict[str, Any], name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _timeout(params: dict[str, Any]) -> float | None:
        value = params.get("timeout")
        if value is None:
            return None
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("timeout must be a non-negative number")
        return float(value)
