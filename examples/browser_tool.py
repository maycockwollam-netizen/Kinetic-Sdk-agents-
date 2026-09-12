"""Run with ``python examples/browser_tool.py`` after ``pip install -e '.[browser]'``."""

import tempfile
from pathlib import Path

from kinetic_sdk.browser import BrowserTool

# Muốn debug sâu hơn với performance trace / network inspection nâng cao, xem
# chrome-devtools-mcp (github.com/ChromeDevTools/chrome-devtools-mcp) — kết nối
# qua Kinetic's MCP client, không phải phần của SDK này.
if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        page = Path(directory) / "page.html"
        page.write_text("<script>console.log('browser ready')</script><h1>Hello</h1>", encoding="utf-8")
        try:
            with BrowserTool() as browser:
                print(browser.execute(action="navigate", url=page.as_uri()).output)
                print(browser.execute(action="get_console_logs").output)
        except ImportError as exc:
            print(exc)
