"""Skill loaders: discover skills from a source, metadata only.

A :class:`SkillLoader` scans one source (a directory of skill folders, a zip
archive, later a git remote) and returns :class:`~kinetic_sdk.skills.skill.Skill`
objects with ``source`` and ``root_path`` set. Implementations must **not**
read the SKILL.md body or any resource file during discovery — only the
frontmatter is parsed. Loading every skill's content up front would waste
context and I/O for skills the agent never uses; the content is read lazily
through :meth:`~kinetic_sdk.skills.skill.Skill.read_main` when the agent
actually picks a skill (progressive disclosure).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from kinetic_sdk.skills.exceptions import SkillParseError
from kinetic_sdk.skills.skill import SKILL_FILE_NAME, Skill

logger = logging.getLogger(__name__)


class SkillLoader(ABC):
    """Interface for discovering skills from a single source.

    Implementations return metadata-only :class:`Skill` objects. They must
    NOT read the SKILL.md body or resource files inside :meth:`discover` —
    that laziness is the progressive-disclosure contract (see the module
    docstring).
    """

    @abstractmethod
    def discover(self) -> list[Skill]:
        """Scan the source and return the discovered skills (metadata only).

        A skill whose SKILL.md fails to parse (missing/invalid frontmatter)
        is logged as a warning and *skipped* — one broken skill must not take
        down the whole scan. A failure of the source itself (e.g. the root
        directory does not exist) may raise, because then the entire source
        is broken, not just one entry.
        """


class FileSystemSkillLoader(SkillLoader):
    """Discover skills from a directory of skill folders.

    Each *direct* subdirectory (no deeper recursion) containing a SKILL.md is
    treated as one skill. Entries that are not directories, and directories
    without a SKILL.md, are skipped silently — an unrelated folder sitting
    next to skills is not an error.

    Args:
        root_dir: Directory containing the skill folders. Must exist —
            raises :class:`ValueError` otherwise, the same contract as
            :class:`~kinetic_sdk.workspace.manager.Workspace`.
        source_label: Value assigned to ``Skill.source`` for every skill from
            this loader (default ``"local"``).
    """

    def __init__(
        self, root_dir: str | os.PathLike[str], source_label: str = "local"
    ) -> None:
        root = os.path.realpath(os.fspath(root_dir))
        if not os.path.isdir(root):
            raise ValueError(f"skill root is not an existing directory: {root_dir!r}")
        self._root = root
        self._source_label = source_label

    @property
    def root_dir(self) -> str:
        """Canonical absolute path of the directory being scanned."""
        return self._root

    def discover(self) -> list[Skill]:
        """Scan :attr:`root_dir` and return one :class:`Skill` per valid folder.

        Results are sorted for determinism. A folder whose SKILL.md fails to
        parse (or whose frontmatter ``name`` does not match the folder name)
        is logged and skipped. Two folders resolving to the same real path
        (via symlinks) yield only one skill — the duplicate is logged and
        skipped so the same content is never registered twice.
        """
        skills: list[Skill] = []
        seen_real_paths: set[str] = set()
        for entry in sorted(os.listdir(self._root)):
            folder = os.path.join(self._root, entry)
            if not os.path.isdir(folder):
                continue
            if not os.path.isfile(os.path.join(folder, SKILL_FILE_NAME)):
                continue
            real = os.path.realpath(folder)
            if real in seen_real_paths:
                logger.warning(
                    "skipping duplicate skill folder %r (same real path as an "
                    "already-discovered skill)",
                    folder,
                )
                continue
            try:
                skill = Skill.from_directory(folder, source=self._source_label)
            except SkillParseError as exc:
                logger.warning("skipping unparsable skill at %r: %s", folder, exc)
                continue
            seen_real_paths.add(real)
            skills.append(skill)
        return skills
