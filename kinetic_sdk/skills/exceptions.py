"""Exceptions raised by the skills package.

Every message names the skill or path involved so the caller (and the audit
trail) can see exactly *what* failed — the same convention
:class:`~kinetic_sdk.secret.registry.SecretNotFoundError` follows.
"""

from __future__ import annotations


class SkillParseError(ValueError):
    """A SKILL.md is missing required frontmatter or is malformed.

    Raised when the ``---`` frontmatter block is absent, a required key
    (``name`` / ``description``) is missing or invalid, or a frontmatter line
    cannot be parsed. The message always names the offending skill/path.
    """


class SkillNotFoundError(KeyError):
    """A registry lookup named a skill that was never accepted.

    The message names the skill that was searched for so callers immediately
    know which name failed, unlike a bare :class:`KeyError`.
    """


class ZipSkillError(ValueError):
    """A skill zip archive is unsafe or unreadable.

    Raised for zip-slip entries (paths escaping the extraction root),
    uncompressed-size or entry-count limits being exceeded (zip-bomb
    protection), or an unreadable archive. The message names the zip and the
    offending entry where applicable.
    """


class SkillVetError(RuntimeError):
    """Security vetting rejected a skill, or vetting was required but absent.

    Raised by :func:`~kinetic_sdk.skills.vet.vet_skill` when a critical flag
    is found and the caller chose the raising policy, and by
    :meth:`~kinetic_sdk.skills.registry.SkillRegistry.add` when a non-local
    skill is offered without a passing
    :class:`~kinetic_sdk.skills.vet.VetResult`. The message names the skill
    and what was missing (no vet result, or which critical category fired).
    """
