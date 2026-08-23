"""Plugin discovery: find candidate plugins WITHOUT importing their code.

Two sources are supported and merged into one ``list[PluginManifest]``:

* **Entry points** — packages installed via ``pip`` register under the
  ``kinetic_sdk.plugins`` group (standard Python packaging mechanism).
* **Directory convention** — every sub-directory of a configured directory
  (e.g. ``~/.kinetic/plugins/``) that contains a ``PLUGIN.md`` is a plugin.

Discovery reads metadata only (entry-point metadata / PLUGIN.md
frontmatter). It NEVER imports plugin modules — listing N plugins must stay
cheap and free of side effects, the same progressive-disclosure rule
:mod:`kinetic_sdk.skills.loader` follows. Import happens exclusively in
:mod:`kinetic_sdk.plugin.loader`, after the static scanner has passed.

A name collision BETWEEN the two sources is a hard error raised here, at
discovery time — silently preferring one source would let a directory
plugin shadow an installed package (or vice versa) without anyone noticing.
Within one source, exact duplicates cannot occur (entry-point names are
unique per environment; directory names are unique per directory).
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable
from typing import Any

from kinetic_sdk.plugin.exceptions import PluginManifestError
from kinetic_sdk.plugin.manifest import (
    ENTRY_POINT_GROUP,
    PLUGIN_FILE_NAME,
    PluginManifest,
)

logger = logging.getLogger(__name__)

#: Something callable that returns entry-point-like objects (``name`` /
#: ``value`` attributes, optionally ``dist.version``). Injectable so tests
#: need no real pip-installed packages.
EntryPointsProvider = Callable[[], Iterable[Any]]


def _default_entry_points() -> Iterable[Any]:
    return importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)


def _entry_point_version(ep: Any) -> str | None:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None
    version = getattr(dist, "version", None)
    return version if isinstance(version, str) else None


def discover_entry_point_plugins(
    entry_points_provider: EntryPointsProvider | None = None,
) -> list[PluginManifest]:
    """Discover plugins registered under the ``kinetic_sdk.plugins`` group.

    Args:
        entry_points_provider: Optional replacement for
            ``importlib.metadata.entry_points`` (tests, alternative
            environments). Called with no arguments; must yield objects with
            ``name`` and ``value`` attributes.
    """
    provider = (
        entry_points_provider if entry_points_provider is not None else _default_entry_points
    )
    manifests: list[PluginManifest] = []
    for ep in provider():
        manifests.append(
            PluginManifest.from_entry_point(
                name=ep.name, value=ep.value, version=_entry_point_version(ep)
            )
        )
    return manifests


def discover_directory_plugins(directory: str | os.PathLike[str]) -> list[PluginManifest]:
    """Discover plugins under *directory* via the PLUGIN.md convention.

    One level deep, sorted for determinism. Sub-directories without a
    PLUGIN.md are skipped silently (not every directory is a plugin); a
    directory WITH a PLUGIN.md that fails to parse is skipped with a
    warning — one malformed plugin must not break discovery of the rest,
    mirroring :class:`~kinetic_sdk.skills.loader.FileSystemSkillLoader`.
    """
    directory = os.path.realpath(os.fspath(directory))
    manifests: list[PluginManifest] = []
    if not os.path.isdir(directory):
        logger.warning("plugin directory %r does not exist; skipping", directory)
        return manifests
    for entry in sorted(os.listdir(directory)):
        subdir = os.path.join(directory, entry)
        if not os.path.isdir(subdir):
            continue
        if not os.path.isfile(os.path.join(subdir, PLUGIN_FILE_NAME)):
            continue
        try:
            manifests.append(PluginManifest.from_directory(subdir))
        except PluginManifestError as exc:
            logger.warning("skipping malformed plugin at %r: %s", subdir, exc)
    return manifests


def discover_plugins(
    directory: str | os.PathLike[str] | None = None,
    entry_points_provider: EntryPointsProvider | None = None,
) -> list[PluginManifest]:
    """Merge both discovery sources into one manifest list.

    Args:
        directory: Plugin directory to scan; ``None`` disables the
            directory source.
        entry_points_provider: Optional entry-points replacement; ``None``
            uses the real ``importlib.metadata`` (an empty group is fine).

    Raises:
        PluginManifestError: two sources produced the same plugin name. The
            message names the plugin and both origins — raised BEFORE
            anything is imported, so neither candidate gains an advantage.
    """
    manifests = discover_entry_point_plugins(entry_points_provider)
    if directory is not None:
        manifests.extend(discover_directory_plugins(directory))
    seen: dict[str, PluginManifest] = {}
    for manifest in manifests:
        previous = seen.get(manifest.name)
        if previous is not None:
            raise PluginManifestError(
                f"plugin name {manifest.name!r} is claimed by both "
                f"{previous.source!r} and {manifest.source!r} discovery — "
                "rename one of them; refusing to pick a winner silently"
            )
        seen[manifest.name] = manifest
    return manifests
