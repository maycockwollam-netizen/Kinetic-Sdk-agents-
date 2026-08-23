"""Exceptions raised by the plugin package.

Every message names the plugin or path involved so the caller (and the audit
trail) can see exactly *what* failed — the same convention the skills and
secret packages follow.
"""

from __future__ import annotations


class PluginManifestError(ValueError):
    """A plugin manifest is missing required fields or is malformed.

    Raised when a PLUGIN.md frontmatter block is absent or unparsable, a
    required key (``name`` / ``entry_point``) is missing or invalid, the
    declared capabilities are unknown, or two discovered plugins (from entry
    points and/or a directory) collide on the same name.
    """


class PluginVetError(RuntimeError):
    """The static scanner rejected a plugin — it was NEVER imported.

    Raised by :meth:`~kinetic_sdk.plugin.loader.PluginLoader.load` when
    :func:`~kinetic_sdk.plugin.scanner.scan_plugin` reports
    ``ScanResult.clean == False``. The message names the plugin and the
    critical finding categories.

    A passing scan is NOT a safety guarantee — see
    :mod:`kinetic_sdk.plugin.scanner` for why the scanner is a tripwire, not
    a sandbox.
    """


class PluginLoadError(RuntimeError):
    """A plugin failed to locate, import, or produce tools.

    Raised for modules that cannot be found without executing code,
    import/factory exceptions (the original exception is chained via
    ``__cause__``), factories returning anything other than ``list[Tool]``,
    unsupported declared capabilities, and tool-name collisions at merge
    time. One broken plugin raises this for itself only — it must never
    crash the loading of the remaining plugins.
    """
