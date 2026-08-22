"""Shared fakes for MCP tests: a scriptable stdio MCP server and an SSE one.

Nothing here talks to a real Unity/Roblox/GitHub server — the fakes speak
just enough JSON-RPC 2.0 / MCP to exercise the SDK's client, transports and
server over real pipes and real sockets.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: A minimal MCP server speaking newline-delimited JSON-RPC on stdio.
#: Supports: initialize, notifications/initialized, tools/list, tools/call
#: with tools "echo" (echoes the text arg), "fail" (MCP tool error) and
#: "crash" (kills the process with exit code 3).
FAKE_MCP_SERVER_SCRIPT = '''
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo back the 'text' argument",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "fail", "description": "Always returns an MCP tool error",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "crash", "description": "Kills the server process",
     "inputSchema": {"type": "object", "properties": {}}},
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        send({"jsonrpc": "2.0", "id": None,
              "error": {"code": -32700, "message": "Parse error"}})
        continue
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.1"},
        }})
    elif method == "notifications/initialized":
        pass  # notification: never answered
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": str(args.get("text", ""))}],
                "isError": False,
            }})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "boom"}],
                "isError": True,
            }})
        elif name == "crash":
            sys.exit(3)
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32602, "message": "unknown tool"}})
    else:
        send({"jsonrpc": "2.0", "id": mid,
              "error": {"code": -32601, "message": "Method not found"}})
'''


def handle_fake_mcp_request(msg: dict) -> dict | None:
    """Pure-function version of the fake server logic (used by the SSE fake).

    Returns the JSON-RPC response dict, or None for notifications.
    """
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp-sse", "version": "0.1"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo back the 'text' argument",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
            },
        }
    if method == "tools/call":
        params = msg.get("params") or {}
        args = params.get("arguments") or {}
        if params.get("name") == "echo":
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "content": [{"type": "text", "text": str(args.get("text", ""))}],
                    "isError": False,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": -32602, "message": "unknown tool"},
        }
    return {
        "jsonrpc": "2.0",
        "id": mid,
        "error": {"code": -32601, "message": "Method not found"},
    }


class FakeSSEMCPServer:
    """A throwaway HTTP server speaking MCP-over-SSE on 127.0.0.1.

    Args:
        behavior: ``"normal"`` (endpoint event + answers every POST),
            ``"silent"`` (accepts the SSE GET but never sends any event, to
            exercise connect/read timeouts), ``"quiet"`` (sends the endpoint
            event but never answers POSTs with SSE messages).
    """

    def __init__(self, behavior: str = "normal") -> None:
        self.behavior = behavior
        self._messages: "queue.Queue[dict]" = queue.Queue()
        self._stopped = threading.Event()
        #: Authorization headers observed on POSTs (for credential tests).
        self.received_auth_headers: list[str | None] = []

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep tests quiet
                pass

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                if self.path.split("?")[0] != "/sse":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if server_ref.behavior == "silent":
                    try:
                        while not server_ref._stopped.is_set():
                            server_ref._stopped.wait(0.05)
                    finally:
                        return
                try:
                    self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
                    self.wfile.flush()
                    while not server_ref._stopped.is_set():
                        try:
                            msg = server_ref._messages.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        payload = json.dumps(msg).encode()
                        self.wfile.write(b"event: message\ndata: " + payload + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                if self.path.split("?")[0] != "/messages":
                    self.send_error(404)
                    return
                server_ref.received_auth_headers.append(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    msg = json.loads(body.decode("utf-8"))
                    response = handle_fake_mcp_request(msg)
                except Exception:  # noqa: BLE001 - fake server
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                if response is not None and server_ref.behavior != "quiet":
                    server_ref._messages.put(response)
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/sse"

    def close(self) -> None:
        self._stopped.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
