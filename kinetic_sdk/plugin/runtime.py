"""Container entrypoint: load a plugin factory and expose its tools as MCP."""
from __future__ import annotations

import importlib
import sys

from kinetic_sdk.mcp.server import MCPServer
from kinetic_sdk.security import PermissivePolicy


def main() -> None:
    module_name, factory_name = sys.argv[1].split(":", 1)
    tools = getattr(importlib.import_module(module_name), factory_name)()
    MCPServer.serve_stdio(tools=tools, permission_policy=PermissivePolicy())
if __name__ == "__main__": main()
