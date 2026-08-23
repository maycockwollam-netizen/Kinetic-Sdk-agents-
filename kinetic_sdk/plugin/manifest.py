"""The :class:`PluginManifest`: metadata for one discoverable plugin.

A directory-convention plugin lives in a directory following this layout::

    <plugin-name>/
    ├── PLUGIN.md       # required — YAML frontmatter (---...---) + Markdown body
    ├── plugin.py       # the module named by ``entry_point`` (any name/package)
    └── ...             # further Python files, all scanned before import

The frontmatter is parsed with the same minimal flat ``key: value`` parser
the skills package uses, so plugin metadata stays dependency-free. The
Markdown *body* is free-form documentation for humans; it is never executed
and never shown to the model automatically.

**Progressive disclosure**, exactly as in
:mod:`kinetic_sdk.skills.skill`: discovery only reads PLUGIN.md frontmatter
(or entry-point metadata) — it NEVER imports plugin code. Importing happens
exclusively in :class:`~kinetic_sdk.plugin.loader.PluginLoader`, AFTER the
static scanner has passed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# The skills frontmatter parser is shared within the SDK on purpose: plugin
# manifests use the exact same flat key: value format as SKILL.md, and
# duplicating the parser would let the two drift apart.
from kinetic_sdk.skills.skill import (
    MAX_NAME_LENGTH,
    SKILL_NAME_PATTERN,
    _split_frontmatter,
)

from kinetic_sdk.plugin.exceptions import PluginManifestError

#: Canonical file every plugin directory must contain.
PLUGIN_FILE_NAME = "PLUGIN.md"

#: importlib.metadata entry-point group external packages register under.
ENTRY_POINT_GROUP = "kinetic_sdk.plugins"

PluginSource = Literal["entry_point", "directory"]

#: Capabilities a plugin may declare. Only ``"tool"`` is acted on in this
#: version; ``"hook"`` is reserved so manifests can already declare intent
#: for a future loader without a format change.
KNOWN_CAPABILITIES: frozenset[str] = frozenset({"tool", "hook"})

_ENTRY_POINT_RE = re.compile(
    r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$"
)


@dataclass(frozen=True)
class PluginManifest:
    """Validated metadata for one plugin — construction NEVER imports code.

    Attributes:
        name: Plugin name; must match
            :data:`~kinetic_sdk.skills.skill.SKILL_NAME_PATTERN` and be at
            most ``MAX_NAME_LENGTH`` chars (same rules as skill names).
        version: Optional free-form version string. Entry-point plugins get
            it from the installed distribution when available; directory
            plugins declare it in PLUGIN.md.
        entry_point: ``"module.submodule:factory_function"`` spec. The
            factory is called with no arguments at load time and must return
            ``list[Tool]`` (or, for entry-point plugins acting as manifest
            providers, another ``PluginManifest`` pointing at the real
            factory).
        source: ``"entry_point"`` or ``"directory"``. Assigned by the
            loader/discovery layer, NEVER read from the plugin's own files,
            so a plugin cannot forge a trusted origin.
        declared_capabilities: What the plugin says it will do (e.g.
            ``{"tool"}``). Must be non-empty and a subset of
            :data:`KNOWN_CAPABILITIES`. Note this is a *declaration*, not an
            enforcement boundary — a plugin is in-process Python code and
            can do anything Python can; the declaration exists so scanning
            and auditing can compare intent against behaviour.
        directory: Plugin root when ``source == "directory"`` (canonicalised
            absolute path); must be ``None`` for entry-point plugins.
    """

    name: str
    version: str | None
    entry_point: str
    source: PluginSource
    declared_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({"tool"})
    )
    directory: str | None = None

    def __post_init__(self) -> None:
        if len(self.name) > MAX_NAME_LENGTH or not SKILL_NAME_PATTERN.match(self.name):
            raise PluginManifestError(
                f"invalid plugin name {self.name!r}: must match "
                f"{SKILL_NAME_PATTERN.pattern!r} and be <= {MAX_NAME_LENGTH} chars"
            )
        if self.version is not None and (
            not isinstance(self.version, str) or not self.version.strip()
        ):
            raise PluginManifestError(
                f"plugin {self.name!r}: version must be a non-empty string or None"
            )
        if not _ENTRY_POINT_RE.match(self.entry_point):
            raise PluginManifestError(
                f"plugin {self.name!r}: invalid entry_point {self.entry_point!r} "
                "(expected 'module.submodule:factory_function')"
            )
        if self.source not in ("entry_point", "directory"):
            raise PluginManifestError(
                f"plugin {self.name!r}: invalid source {self.source!r} "
                "(expected 'entry_point' or 'directory')"
            )
        capabilities = frozenset(self.declared_capabilities)
        if not capabilities:
            raise PluginManifestError(
                f"plugin {self.name!r}: declared_capabilities must be non-empty"
            )
        unknown = capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise PluginManifestError(
                f"plugin {self.name!r}: unknown capabilities {sorted(unknown)} "
                f"(known: {sorted(KNOWN_CAPABILITIES)})"
            )
        object.__setattr__(self, "declared_capabilities", capabilities)
        if self.source == "directory":
            if not self.directory:
                raise PluginManifestError(
                    f"plugin {self.name!r}: source 'directory' requires a directory path"
                )
            object.__setattr__(
                self, "directory", os.path.realpath(os.fspath(self.directory))
            )
        elif self.directory is not None:
            raise PluginManifestError(
                f"plugin {self.name!r}: directory must be None for entry-point plugins"
            )

    @property
    def module_spec(self) -> str:
        """The module part of ``entry_point`` (text before the colon)."""
        return self.entry_point.split(":", 1)[0]

    @property
    def factory_name(self) -> str:
        """The callable part of ``entry_point`` (text after the colon)."""
        return self.entry_point.split(":", 1)[1]

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str]) -> "PluginManifest":
        """Build a manifest from a plugin directory's PLUGIN.md frontmatter.

        Only the frontmatter is read — the Markdown body and every Python
        file stay untouched (progressive disclosure: discovery must not
        import or execute plugin code).

        Args:
            directory: The plugin directory. Its base name must equal the
                ``name`` declared in frontmatter (same rule as skills: a
                mismatch means the package was renamed or tampered with).

        Raises:
            PluginManifestError: PLUGIN.md missing, frontmatter malformed,
                a required field missing/invalid, or the frontmatter ``name``
                does not match the directory name.
        """
        directory = os.path.realpath(os.fspath(directory))
        plugin_md = os.path.join(directory, PLUGIN_FILE_NAME)
        if not os.path.isfile(plugin_md):
            raise PluginManifestError(f"{directory}: missing required {PLUGIN_FILE_NAME}")
        text = Path(plugin_md).read_text(encoding="utf-8")
        try:
            frontmatter, _body = _split_frontmatter(text, plugin_md)
        except ValueError as exc:
            raise PluginManifestError(str(exc)) from exc
        name = frontmatter.get("name", "")
        dirname = os.path.basename(directory)
        if name != dirname:
            raise PluginManifestError(
                f"{plugin_md}: frontmatter name {name!r} does not match "
                f"directory name {dirname!r}"
            )
        entry_point = frontmatter.get("entry_point", "")
        if not entry_point:
            raise PluginManifestError(
                f"{plugin_md}: missing required 'entry_point' key"
            )
        capabilities = frontmatter.get("capabilities", "tool")
        return cls(
            name=name,
            version=frontmatter.get("version") or None,
            entry_point=entry_point,
            source="directory",
            declared_capabilities=frozenset(
                part.strip() for part in capabilities.split(",") if part.strip()
            ),
            directory=directory,
        )

    @classmethod
    def from_entry_point(
        cls, name: str, value: str, version: str | None = None
    ) -> "PluginManifest":
        """Build a manifest from an importlib.metadata entry point.

        The entry point's *name* becomes the plugin name and its *value*
        (``"module:factory"``) the entry_point spec. Capabilities default to
        ``{"tool"}`` because they cannot be read without importing the
        package — and discovery must not import. ``"tool"`` is the only
        capability this version acts on, so nothing is lost.
        """
        return cls(
            name=name,
            version=version,
            entry_point=value,
            source="entry_point",
            declared_capabilities=frozenset({"tool"}),
        )
