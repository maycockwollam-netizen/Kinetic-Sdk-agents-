"""The skill registry: the single gatekeeper for accepted skills.

A :class:`SkillRegistry` never auto-accepts whatever a loader returns.
Skills enter only through :meth:`SkillRegistry.add` /
:meth:`SkillRegistry.add_many` (or the :meth:`SkillRegistry.load_from`
convenience wrapper), and the registry enforces the central invariant in
exactly one place: **a skill whose ``source`` is not ``"local"`` may only be
accepted together with a passing**
:class:`~kinetic_sdk.skills.vet.VetResult`. There is deliberately no way
around this — no constructor that pre-registers loaders, no "trusted bulk
import" flag.

Name collisions follow *first registration wins*: re-adding an existing name
logs a warning and keeps the original. That makes re-scanning a directory a
safe no-op rather than an error or a silent overwrite.
"""

from __future__ import annotations

import logging

from kinetic_sdk.skills.exceptions import SkillNotFoundError, SkillVetError
from kinetic_sdk.skills.loader import SkillLoader
from kinetic_sdk.skills.skill import Skill
from kinetic_sdk.skills.vet import VetResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Holds accepted skills and enforces the vetting invariant.

    Local skills (``source == "local"``) are trusted and added directly.
    Anything else (``"zip"``, a future ``"github"``, ...) must come with a
    :class:`~kinetic_sdk.skills.vet.VetResult` whose ``clean`` is ``True`` —
    see :meth:`add`.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    # --- acceptance --------------------------------------------------------

    def add(self, skill: Skill, vet_result: VetResult | None = None) -> None:
        """Accept one skill into the registry.

        Args:
            skill: The skill to accept. Duplicate names lose to the skill
                that was registered first (warning logged, not an error).
            vet_result: Required for non-local skills: the result of
                :func:`~kinetic_sdk.skills.vet.vet_skill`. Ignored for local
                skills, which are trusted by definition.

        Raises:
            SkillVetError: ``skill.source != "local"`` and *vet_result* is
                missing or ``clean`` is ``False``. The message names the
                skill and what was missing — this is the one choke point of
                the "no unvetted external skill" invariant.
        """
        if skill.source != "local":
            if vet_result is None:
                raise SkillVetError(
                    f"skill {skill.name!r} (source={skill.source!r}) cannot be "
                    "accepted without a VetResult — non-local skills must be "
                    "vetted first"
                )
            if not vet_result.clean:
                categories = sorted({f.category for f in vet_result.flags})
                raise SkillVetError(
                    f"skill {skill.name!r} (source={skill.source!r}) failed "
                    f"vetting (flags: {categories}) and cannot be accepted"
                )
        if skill.name in self._skills:
            logger.warning(
                "skill %r is already registered; keeping the first registration "
                "and ignoring the new one from source %r",
                skill.name,
                skill.source,
            )
            return
        self._skills[skill.name] = skill

    def add_many(
        self, skills: list[Skill], vet_result: VetResult | None = None
    ) -> None:
        """Add several skills sharing ONE :class:`VetResult`.

        Convenience for sources (e.g. one zip archive) that were vetted as a
        single unit. If skills need individual vet outcomes, call :meth:`add`
        per skill with its own result instead.
        """
        for skill in skills:
            self.add(skill, vet_result=vet_result)

    def load_from(
        self, loader: SkillLoader, vet_result: VetResult | None = None
    ) -> list[Skill]:
        """Discover via *loader* and accept everything it returns.

        Suited to local :class:`FileSystemSkillLoader` sources where
        ``vet_result=None`` is valid (their source is ``"local"``). For a
        :class:`ZipSkillLoader`, callers must vet each skill *before* adding
        — use :func:`~kinetic_sdk.skills.add_skill_from_zip` instead of
        calling this on an unvetted zip loader.
        """
        skills = loader.discover()
        self.add_many(skills, vet_result=vet_result)
        return skills

    # --- access --------------------------------------------------------------

    def list_skills(self) -> list[Skill]:
        """Return every accepted skill (metadata only, no extra I/O)."""
        return list(self._skills.values())

    def get(self, name: str) -> Skill:
        """Return the skill registered under *name*.

        Raises:
            SkillNotFoundError: no skill with that name was accepted; the
                message names what was searched for.
        """
        try:
            return self._skills[name]
        except KeyError:
            raise SkillNotFoundError(
                f"no skill named {name!r} is registered"
            ) from None

    def load(self, name: str) -> str:
        """Read the full content of a registered skill.

        Named ``load`` — not ``invoke`` — because a skill is instructional
        *text* that is read, not an action that is executed; "invoke" would
        wrongly suggest a tool-like side effect.
        """
        return self.get(name).read_main()

    def to_prompt_catalog(self) -> str:
        """Format a compact ``name: description`` catalogue for a prompt.

        One ``- {name}: {description}`` line per skill, sorted by name so the
        output is deterministic and cheap to inject into a system prompt or
        tool description later.
        """
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in sorted(self._skills.values(), key=lambda s: s.name)
        )
