"""Discover skills from a user-supplied ``.zip`` archive, safely.

The archive is extracted into a temporary directory that **lives as long as
the loader instance** — not merely for the duration of :meth:`discover` —
and parsing is delegated to
:class:`~kinetic_sdk.skills.loader.FileSystemSkillLoader` so SKILL.md parsing
logic exists in exactly one place.

**Lifecycle contract (binding).** The temp dir is created on the first
:meth:`discover` call and removed only by :meth:`close` (or by leaving a
``with`` block). The :class:`~kinetic_sdk.skills.skill.Skill` objects
returned point their ``root_path`` into this temp dir, so the caller MUST
keep the loader alive for as long as it still reads skill content::

    with ZipSkillLoader(zip_path) as loader:
        skills = loader.discover()
        for skill in skills:
            body = skill.read_main()   # fine — temp dir alive
    # past the `with`, skill.read_main() raises SkillParseError

(Callers that want skills to outlive the loader — e.g. anything feeding a
long-lived :class:`~kinetic_sdk.skills.registry.SkillRegistry` — should use
:func:`~kinetic_sdk.skills.add_skill_from_zip`, which *materialises* vetted
skills into a persistent directory instead of handing out temp-dir-backed
skills.)

Safety rules (mandatory, no opt-out flags):

* **Two independent containment layers, never merged.** (a) Here, zip entries
  are just unverified strings and no ``Skill``/``Workspace`` exists yet, so
  the extraction root cannot be a ``Workspace`` (chicken-and-egg): each entry
  is canonicalised and ``commonpath``-checked against the temp dir by hand.
  (b) Later, :meth:`~kinetic_sdk.skills.skill.Skill.read_resource` re-checks
  reads through a real ``Workspace``. Layer (a) guards extraction, layer (b)
  guards agent-time reads; neither replaces the other.
* **Zip-slip**: any entry whose resolved destination escapes the temp dir
  raises :class:`ZipSkillError` — a traversal attempt poisons the whole
  archive.
* **Symlinks are always refused.** No attempt is made to allow "safe"
  symlinks (deciding whether a target escapes is a classic source of bugs).
  A symlink entry is *skipped* — never extracted, no trace left on disk, and
  extraction of the remaining valid entries continues; a warning names the
  entry. Same policy as unknown extensions: one suspicious entry should not
  block the valid skills sharing the archive.
* **Zip bombs**: the sum of declared uncompressed sizes and the entry count
  are checked *before* anything is written, and bytes actually written are
  counted during extraction as a second guard against forged headers.
  Exceeding either limit raises :class:`ZipSkillError` immediately.
* **Extension allowlist** applies to *files only* — directory entries always
  pass, otherwise ``scripts/``/``references/``/``assets/`` (which have no
  extension) would be filtered out. A file with a disallowed extension is
  skipped with a warning, not an error for the whole archive.

``ZipFile.extractall()`` is never used: entries are validated one by one and
written via :meth:`ZipFile.open` to the already-validated destination.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import zipfile

from kinetic_sdk.skills.exceptions import ZipSkillError
from kinetic_sdk.skills.loader import FileSystemSkillLoader, SkillLoader
from kinetic_sdk.skills.skill import Skill

logger = logging.getLogger(__name__)

#: Default file extensions permitted in a skill archive. Directories are
#: always permitted; this filters files only.
DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".txt", ".json", ".py", ".sh", ".yaml", ".yml"}
)

_CHUNK_SIZE = 64 * 1024


class ZipSkillLoader(SkillLoader):
    """Discover skills from a ``.zip`` file, extracting it safely first.

    Args:
        zip_path: Path to the archive. Must be an existing file.
        max_total_size_bytes: Cap on the total *uncompressed* size of all
            entries (zip-bomb protection). Exceeding it raises
            :class:`ZipSkillError` before extraction completes.
        max_entries: Cap on the number of archive entries. Exceeding it
            raises :class:`ZipSkillError`.
        allowed_extensions: File extensions (lowercase, with dot) permitted
            in the archive; ``None`` uses :data:`DEFAULT_ALLOWED_EXTENSIONS`.
            Directory entries are never filtered by this list.

    The loader is a context manager; see the module docstring for the
    lifecycle contract that ties the skills it returns to its temp dir.
    """

    def __init__(
        self,
        zip_path: str | os.PathLike[str],
        max_total_size_bytes: int = 10 * 1024 * 1024,
        max_entries: int = 500,
        allowed_extensions: frozenset[str] | None = None,
    ) -> None:
        path = os.path.realpath(os.fspath(zip_path))
        if not os.path.isfile(path):
            raise ValueError(f"skill zip is not an existing file: {zip_path!r}")
        self._zip_path = path
        self._max_total_size = max_total_size_bytes
        self._max_entries = max_entries
        self._allowed_extensions = (
            allowed_extensions
            if allowed_extensions is not None
            else DEFAULT_ALLOWED_EXTENSIONS
        )
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._skills: list[Skill] | None = None

    # --- SkillLoader -----------------------------------------------------

    def discover(self) -> list[Skill]:
        """Extract (once) and return the skills inside the archive.

        The first call creates the temp dir, extracts the archive safely,
        and delegates parsing to
        :class:`~kinetic_sdk.skills.loader.FileSystemSkillLoader` with
        ``source_label="zip"``. Later calls return the cached result without
        re-extracting.

        The returned skills stay readable only while this loader is open —
        see the module docstring.

        Raises:
            RuntimeError: the loader has already been closed.
            ZipSkillError: the archive is unreadable, contains a zip-slip
                entry, or exceeds the size/entry limits.
        """
        if self._closed:
            raise RuntimeError(
                f"ZipSkillLoader for {self._zip_path!r} is closed; "
                "discover() can no longer extract"
            )
        if self._skills is not None:
            return list(self._skills)
        try:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="kinetic-skills-")
            self._extract_safely(self._temp_dir.name)
            self._skills = FileSystemSkillLoader(
                self._temp_dir.name, source_label="zip"
            ).discover()
        except Exception:
            self.close()
            raise
        return list(self._skills)

    # --- lifecycle ---------------------------------------------------------

    _closed: bool = False

    def close(self) -> None:
        """Remove the temp dir. Idempotent — safe to call more than once."""
        self._closed = True
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> "ZipSkillLoader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup, like StdioTransport
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    # --- safe extraction ----------------------------------------------------

    def _extract_safely(self, dest_root: str) -> None:
        """Validate every entry, then write the allowed ones into *dest_root*."""
        try:
            archive = zipfile.ZipFile(self._zip_path)
        except zipfile.BadZipFile as exc:
            raise ZipSkillError(f"{self._zip_path!r} is not a valid zip: {exc}") from exc
        with archive:
            infos = archive.infolist()
            if len(infos) > self._max_entries:
                raise ZipSkillError(
                    f"{self._zip_path!r}: {len(infos)} entries exceeds the "
                    f"limit of {self._max_entries} (zip-bomb protection)"
                )
            declared_total = sum(info.file_size for info in infos)
            if declared_total > self._max_total_size:
                raise ZipSkillError(
                    f"{self._zip_path!r}: declared uncompressed size "
                    f"{declared_total} bytes exceeds the limit of "
                    f"{self._max_total_size} (zip-bomb protection)"
                )
            written_total = 0
            for info in infos:
                written_total += self._extract_entry(archive, info, dest_root)
                if written_total > self._max_total_size:
                    raise ZipSkillError(
                        f"{self._zip_path!r}: actual uncompressed size exceeds "
                        f"the limit of {self._max_total_size} bytes while "
                        f"extracting {info.filename!r} (forged size header?)"
                    )

    def _extract_entry(
        self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, dest_root: str
    ) -> int:
        """Extract one entry if it passes every check; return bytes written."""
        name = info.filename
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            logger.warning(
                "%r: skipping symlink entry %r (symlinks are never extracted)",
                self._zip_path,
                name,
            )
            return 0

        dest = os.path.realpath(os.path.join(dest_root, name))
        try:
            contained = os.path.commonpath([dest_root, dest]) == dest_root
        except ValueError:
            contained = False
        if not contained:
            raise ZipSkillError(
                f"{self._zip_path!r}: entry {name!r} resolves outside the "
                "extraction directory (zip-slip attempt)"
            )

        if info.is_dir() or name.endswith("/"):
            os.makedirs(dest, exist_ok=True)
            return 0

        extension = os.path.splitext(name)[1].lower()
        if extension not in self._allowed_extensions:
            logger.warning(
                "%r: skipping entry %r (extension %r not in the allowlist)",
                self._zip_path,
                name,
                extension,
            )
            return 0

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        written = 0
        with archive.open(info) as source, open(dest, "wb") as target:
            shutil.copyfileobj(source, target, length=_CHUNK_SIZE)
            written = target.tell()
        return written
