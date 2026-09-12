"""Skills package: reusable Markdown instruction packages with vetting.

A *skill* is a directory with a ``SKILL.md`` (YAML frontmatter + Markdown
body) and optional ``scripts/`` / ``references/`` / ``assets/`` resources.
Skills are discovered metadata-first (progressive disclosure), and skills
from untrusted sources must pass :func:`~kinetic_sdk.skills.vet.vet_skill`
before :class:`~kinetic_sdk.skills.registry.SkillRegistry` will accept them.

Typical flows::

    # Local, trusted directory:
    registry = SkillRegistry()
    registry.load_from(FileSystemSkillLoader("./skills"))
    print(registry.to_prompt_catalog())
    body = registry.load("code-review")

    # User-uploaded zip (vetted, then materialised to persistent storage):
    accepted = add_skill_from_zip(registry, "upload.zip", persist_dir="./skills")

Loading a skill NEVER grants tools or permissions by itself: a skill is
instructional text only, and anything it tells the agent to do still goes
through the agent's ``permission_policy`` like any other model output.
"""

from __future__ import annotations

import logging
import os
import shutil

from kinetic_sdk.skills.activation import SkillActivationHook, select_active_skills
from kinetic_sdk.skills.exceptions import (
    SkillNotFoundError,
    SkillParseError,
    SkillVetError,
    ZipSkillError,
)
from kinetic_sdk.skills.loader import FileSystemSkillLoader, SkillLoader
from kinetic_sdk.skills.registry import SkillRegistry
from kinetic_sdk.skills.skill import Skill
from kinetic_sdk.skills.vet import (
    LLMSkillReviewer,
    StaticSkillScanner,
    VetFlag,
    VetResult,
    vet_skill,
)
from kinetic_sdk.skills.zip_loader import ZipSkillLoader
from kinetic_sdk.workspace.manager import Workspace

logger = logging.getLogger(__name__)

__all__ = [
    "FileSystemSkillLoader",
    "LLMSkillReviewer",
    "Skill",
    "SkillActivationHook",
    "SkillLoader",
    "SkillNotFoundError",
    "SkillParseError",
    "SkillRegistry",
    "SkillVetError",
    "StaticSkillScanner",
    "VetFlag",
    "VetResult",
    "ZipSkillError",
    "ZipSkillLoader",
    "add_skill_from_zip",
    "select_active_skills",
    "vet_skill",
]


def add_skill_from_zip(
    registry: SkillRegistry,
    zip_path: str | os.PathLike[str],
    persist_dir: str | os.PathLike[str],
    llm_reviewer: LLMSkillReviewer | None = None,
    raise_on_critical: bool = True,
) -> list[Skill]:
    """Discover, vet, and ACCEPT the skills inside one zip archive.

    This helper is where the zip↔skill lifecycle is bound correctly: every
    skill that passes vetting is **materialised** — copied from the loader's
    temp dir to ``persist_dir/<skill-name>/`` — *while the temp dir is still
    alive*, and only the persistent
    :class:`~kinetic_sdk.skills.skill.Skill` (``root_path`` under
    *persist_dir*, ``source`` still ``"zip"``) is handed to the registry.
    The registry therefore never holds a skill pointing into a temp dir, and
    never needs to know a :class:`ZipSkillLoader` existed. The temp-dir-backed
    discovery objects are used only inside this function, for vetting and
    copying.

    Args:
        registry: The registry to accept the vetted skills into.
        zip_path: Path to the uploaded archive.
        persist_dir: Directory where accepted skills live permanently. The
            caller picks a sensible place (e.g. a folder inside the agent's
            workspace); the SDK deliberately does not write to hidden global
            locations like ``~/.kinetic/...``. Created when missing.
        llm_reviewer: Optional independent judge model, forwarded to
            :func:`~kinetic_sdk.skills.vet.vet_skill`.
        raise_on_critical: Forwarded to
            :func:`~kinetic_sdk.skills.vet.vet_skill`. With the default
            ``True`` the first critical skill raises and later skills are not
            processed (earlier ones stay accepted). With ``False`` a
            critical-flagged skill is skipped with a warning instead — note
            this still never lets a failed skill into the registry, per the
            ``vet_skill`` contract.

    Returns:
        The accepted skills, pointing at their persistent copies.

    Raises:
        SkillVetError: a skill failed vetting and ``raise_on_critical=True``.
        FileExistsError: ``persist_dir/<skill-name>/`` already exists — the
            caller decides whether to delete the old copy first; silently
            overwriting a trusted skill would be a downgrade-attack vector.
    """
    os.makedirs(persist_dir, exist_ok=True)
    persist_workspace = Workspace(persist_dir)
    accepted: list[Skill] = []
    with ZipSkillLoader(zip_path) as loader:
        for discovered in loader.discover():
            vet_result = vet_skill(
                discovered,
                llm_reviewer=llm_reviewer,
                raise_on_critical=raise_on_critical,
            )
            if not vet_result.clean:
                logger.warning(
                    "skill %r from %r failed vetting; skipped",
                    discovered.name,
                    zip_path,
                )
                continue
            # The destination passes through the persist-dir Workspace's
            # containment check, exactly like any other path the SDK writes.
            destination = persist_workspace.resolve(discovered.name)
            if os.path.exists(destination):
                raise FileExistsError(
                    f"cannot accept skill {discovered.name!r}: {destination!r} "
                    "already exists — remove the old copy first if you meant "
                    "to update it"
                )
            shutil.copytree(discovered.root_path, destination)
            persistent = Skill.from_directory(destination, source=discovered.source)
            registry.add(persistent, vet_result=vet_result)
            accepted.append(persistent)
    return accepted
