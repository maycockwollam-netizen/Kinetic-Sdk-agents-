"""Plugins: discover + dynamically load external Python packages (Stage 4).

A *plugin* is external Python code that registers additional
:class:`~kinetic_sdk.tool.base.Tool` objects into Kinetic at startup. This
is deliberately NOT an "everything is a plugin" architecture: the agent
loop, ``permission_policy`` and ``EventBus`` are untouched core — plugins
only contribute tools, which then flow through the exact same permission
checks as internal and MCP tools.

**Security model — read this before trusting a scan result.** A plugin runs
IN THE SAME PROCESS as Kinetic. Unlike an MCP server (isolated subprocess —
worst case, Kinetic disconnects) or a skill (inert Markdown — the agent
still chooses whether to follow it), a malicious plugin can do anything
ordinary Python can: read ``os.environ``, open sockets, or monkey-patch
``kinetic_sdk.security`` to disable permission checks for every request
that comes after it — no exploit required.

Python has no real in-process sandbox, so the static scanner
(:mod:`kinetic_sdk.plugin.scanner`) is a TRIPWIRE, not a sandbox: it catches
accidental violations and unsophisticated attacks and raises the cost of
sophisticated ones, but a determined attacker can evade pattern matching
(obfuscation, runtime string decoding, ...). "Plugin passed scan" NEVER
means "plugin is safe". Only load plugins from sources you would trust with
your own credentials.

Typical flow::

    from kinetic_sdk.plugin import PluginRegistry

    registry = PluginRegistry(directory="~/.kinetic/plugins")
    tools = registry.load_all()  # skip-and-log broken plugins by default
    agent = Agent(llm=..., tools=[*tools, GitTool()], ...)

Every load attempt (pass AND fail) is written to the audit log with the
scan findings and registered tool names — the evidence trail for "which
plugin brought tool X into the system".
"""

from kinetic_sdk.plugin.discovery import (
    discover_directory_plugins,
    discover_entry_point_plugins,
    discover_plugins,
)
from kinetic_sdk.plugin.docker_loader import DockerPluginLoader
from kinetic_sdk.plugin.exceptions import (
    PluginLoadError,
    PluginManifestError,
    PluginVetError,
)
from kinetic_sdk.plugin.loader import PluginLoader
from kinetic_sdk.plugin.manifest import (
    ENTRY_POINT_GROUP,
    KNOWN_CAPABILITIES,
    PLUGIN_FILE_NAME,
    PluginManifest,
)
from kinetic_sdk.plugin.registry import PluginRegistry
from kinetic_sdk.plugin.scanner import (
    ScanResult,
    StaticPluginScanner,
    scan_plugin,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "KNOWN_CAPABILITIES",
    "PLUGIN_FILE_NAME",
    "PluginLoader",
    "DockerPluginLoader",
    "PluginLoadError",
    "PluginManifest",
    "PluginManifestError",
    "PluginRegistry",
    "PluginVetError",
    "ScanResult",
    "StaticPluginScanner",
    "discover_directory_plugins",
    "discover_entry_point_plugins",
    "discover_plugins",
    "scan_plugin",
]
