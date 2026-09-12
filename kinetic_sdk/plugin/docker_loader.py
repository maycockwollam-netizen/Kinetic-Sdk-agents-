"""Docker-isolated plugin loader.

This is deliberately an opt-in companion to :mod:`plugin.loader`: AST
scanning remains the cheap defence-in-depth default for in-process plugins.
For ``manifest.isolation == 'docker'`` this module builds a small image and
speaks MCP over ``docker run -i``.  Reusing MCP is a settled decision: it
preserves request correlation, deadlines and untrusted-output redaction
without inventing a second RPC protocol.  The returned adapters are normal
Kinetic tools, so the host Agent still applies its permission policy and
audit log before any container call.

Docker cleanup is best effort.  A cidfile lets ``close`` run ``docker rm -f``
when a host process dies between startup and normal teardown; an OS crash or
unavailable Docker daemon can still leave Docker's normal daemon-side state.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from kinetic_sdk.mcp.adapter import MCPToolAdapter
from kinetic_sdk.mcp.client import MCPClient
from kinetic_sdk.mcp.transport import ProcessFactory, StdioTransport
from kinetic_sdk.plugin.exceptions import PluginLoadError
from kinetic_sdk.plugin.manifest import PluginManifest

DOCKERFILE = '''FROM python:3.12-slim\nWORKDIR /plugin\nCOPY . /plugin\nRUN pip install --no-cache-dir kinetic-agent-sdk\nENTRYPOINT ["python", "-m", "kinetic_sdk.plugin.runtime"]\n'''


class DockerPluginLoader:
    """Build and connect one directory plugin as an MCP subprocess in Docker."""

    def __init__(self, *, docker: str = "docker", runner: Callable[..., object] = subprocess.run,
                 process_factory: ProcessFactory | None = None) -> None:
        self._docker, self._runner, self._process_factory = docker, runner, process_factory
        self._client: MCPClient | None = None
        self._cidfile: str | None = None

    def load(self, manifest: PluginManifest) -> list[MCPToolAdapter]:
        if manifest.isolation != "docker":
            raise PluginLoadError(f"plugin {manifest.name!r} is not configured for docker isolation")
        assert manifest.directory is not None
        image = f"kinetic-plugin-{manifest.name}:{(manifest.version or 'latest').replace('/', '-') }"
        self._run([self._docker, "build", "-t", image, manifest.directory], "build")
        fd, cidfile = tempfile.mkstemp(prefix="kinetic-plugin-", suffix=".cid")
        os.close(fd)
        self._cidfile = cidfile
        args = ["run", "-i", "--cidfile", cidfile, "--read-only", "--network", manifest.network or "none",
                "--memory", manifest.memory_limit, "--cpus", str(manifest.cpu_limit), image, manifest.entry_point]
        try:
            transport = StdioTransport(self._docker, args, process_factory=self._process_factory)
            self._client = MCPClient(transport)
            self._client.initialize()
            return [MCPToolAdapter.from_mcp_schema(self._client, manifest.name, item)
                    for item in self._client.list_tools()]
        except Exception as exc:
            self.close()
            raise PluginLoadError(f"docker plugin {manifest.name!r} failed to start: {exc}") from exc

    def _run(self, argv: list[str], operation: str) -> None:
        completed = self._runner(argv, capture_output=True, text=True)
        if getattr(completed, "returncode", 1) != 0:
            stderr = str(getattr(completed, "stderr", ""))[-500:]
            raise PluginLoadError(f"docker plugin {operation} failed (exit {getattr(completed, 'returncode', '?')}): {stderr}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close(); self._client = None
        if self._cidfile is not None:
            try:
                cid = Path(self._cidfile).read_text().strip()
                if cid:
                    self._runner([self._docker, "rm", "-f", cid], capture_output=True, text=True)
            except OSError:
                pass
            finally:
                try: os.unlink(self._cidfile)
                except OSError: pass
                self._cidfile = None

    def __enter__(self) -> "DockerPluginLoader": return self
    def __exit__(self, *exc_info: object) -> None: self.close()
    def __del__(self) -> None: self.close()
