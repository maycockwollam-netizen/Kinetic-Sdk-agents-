"""A stateful, optional Playwright browser tool for agent workflows."""

from __future__ import annotations

import base64
import time
from typing import Any

from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.workspace.base import WorkspaceBase


class BrowserTool(Tool):
    """Control one headless Playwright page through a compact action schema.

    Playwright is imported only when the first action is executed, keeping the
    core SDK dependency-free.  ``DockerWorkspace`` is accepted to keep the
    workspace composition API uniform; its commands remain the sandbox
    boundary, while the Playwright driver connects from the agent process.
    Deployments that require the browser process itself in the container
    should install the browser extra in that container and expose its CDP
    endpoint to the agent process.
    """

    name = "browser"
    description = "Navigate and interact with a headless web page using CSS selectors."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["navigate", "click", "type", "screenshot", "get_console_logs", "get_network_requests", "wait_for_selector"]},
            "url": {"type": "string"},
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "timeout": {"type": "number", "minimum": 0},
        },
        "required": ["action"],
    }

    def __init__(self, workspace: WorkspaceBase | None = None) -> None:
        self.workspace = workspace
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._console_logs: list[dict[str, str]] = []
        self._network_requests: list[dict[str, Any]] = []
        self._request_started: dict[int, float] = {}

    def __enter__(self) -> "BrowserTool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the page/browser and stop Playwright (safe to call twice)."""
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._page = self._browser = self._playwright = None

    def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "BrowserTool requires Playwright. Install it with: "
                "pip install kinetic-agent-sdk[browser]"
            ) from exc
        # DockerWorkspace has already centralised command sandboxing.  The
        # driver API is stateful, so it must stay in this process to preserve
        # navigation/cookies across tool actions rather than spawning one
        # short-lived docker exec for every action.
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page()
        self._page.on("console", self._record_console)
        self._page.on("request", self._record_request_start)
        self._page.on("response", self._record_response)
        return self._page

    def _record_console(self, message: Any) -> None:
        self._console_logs.append({"type": message.type, "text": message.text})

    def _record_request_start(self, request: Any) -> None:
        self._request_started[id(request)] = time.monotonic()

    def _record_response(self, response: Any) -> None:
        request = response.request
        started = self._request_started.pop(id(request), time.monotonic())
        self._network_requests.append(
            {"url": request.url, "method": request.method, "status": response.status,
             "duration": round(time.monotonic() - started, 4)}
        )

    def execute(self, action: str, **params: Any) -> ToolResult:  # type: ignore[override]
        """Run one browser action and return a recoverable ``ToolResult``."""
        try:
            page = self._ensure_page()
            if action == "navigate":
                url = self._required_string(params, "url")
                page.goto(url)
                return ToolResult(output=f"navigated to {url}")
            if action == "click":
                page.click(self._required_string(params, "selector"), timeout=self._timeout(params))
                return ToolResult(output="click completed")
            if action == "type":
                page.fill(self._required_string(params, "selector"), self._required_string(params, "text"), timeout=self._timeout(params))
                return ToolResult(output="text entered")
            if action == "screenshot":
                image = page.screenshot(type="png")
                return ToolResult(output="screenshot captured", metadata={"image_base64": base64.b64encode(image).decode("ascii"), "media_type": "image/png"})
            if action == "get_console_logs":
                return ToolResult(output=list(self._console_logs))
            if action == "get_network_requests":
                return ToolResult(output=list(self._network_requests))
            if action == "wait_for_selector":
                page.wait_for_selector(self._required_string(params, "selector"), timeout=self._timeout(params))
                return ToolResult(output="selector found")
            return ToolResult(error=f"unsupported browser action: {action!r}")
        except ImportError:
            raise
        except Exception as exc:  # Playwright errors are recoverable tool failures.
            return ToolResult(error=f"browser {action} failed: {type(exc).__name__}: {exc}")

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
