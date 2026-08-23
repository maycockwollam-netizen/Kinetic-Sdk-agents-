"""The :class:`Skill` dataclass: metadata plus lazy content accessors.

A skill on disk is a directory following the Agent Skills layout::

    <skill-name>/
    ├── SKILL.md        # required — YAML frontmatter (---...---) + Markdown body
    ├── scripts/        # optional — never read during discovery
    ├── references/     # optional
    └── assets/         # optional

**Progressive disclosure** is the organising principle: discovery (see
:mod:`kinetic_sdk.skills.loader`) only parses the SKILL.md *frontmatter* —
name, description, optional version — and never touches the Markdown body or
resource files, because cataloguing N skills must stay cheap and must not
flood the agent's context. The full body and resources are read only when a
caller explicitly asks via :meth:`Skill.read_main` / :meth:`Skill.read_resource`.
In other words: *construction/discovery never reads content, but the on-demand
read methods fully may* — that lazy read is the mechanism, not a violation.

Frontmatter is parsed with a minimal hand-rolled parser (flat ``key: value``
lines between ``---`` fences). Skill metadata is intentionally simple, so a
``pyyaml`` dependency is not justified; anything more complex than a flat
mapping raises :class:`SkillParseError` instead of being silently
misinterpreted. If richer frontmatter is ever genuinely needed, ``pyyaml``
should be added as an *optional* extra (the way ``litellm`` is), not a core
dependency.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from kinetic_sdk.skills.exceptions import SkillParseError
from kinetic_sdk.workspace.manager import PathTraversalError, Workspace

logger = logging.getLogger(__name__)

#: Canonical file every skill directory must contain.
SKILL_FILE_NAME = "SKILL.md"

#: Skill names are lowercase alphanumeric words joined by single hyphens.
#: This rules out ``a--b``, leading/trailing hyphens, and uppercase — the
#: stricter shape keeps names usable as directory names, prompt tokens, and
#: registry keys without escaping surprises.
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

#: Maximum length of a skill ``name``.
MAX_NAME_LENGTH = 64

#: Maximum length of a skill ``description``. Longer descriptions are
#: truncated (with a notice) rather than rejected — the description sits in
#: the cheap catalogue shown to the model for *every* skill at once, so one
#: bloated entry must not inflate the catalogue, but a verbose author should
#: not lose the whole skill over it either.
MAX_DESCRIPTION_LENGTH = 1024

#: The only sub-directories whose contents are exposed as resources.
#: Anything else sitting in the skill root (besides SKILL.md) exists on disk
#: but is deliberately invisible through the Skill API — see
#: :meth:`Skill.list_resources` for why this is a separate layer from the zip
#: loader's extension allowlist.
RESOURCE_DIRS: tuple[str, ...] = ("scripts", "references", "assets")

ResourceType = Literal["scripts", "references", "assets"]

_FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")


def _split_frontmatter(text: str, path: str) -> tuple[dict[str, str], str]:
    """Split *text* into ``(frontmatter, body)``.

    The frontmatter is a flat ``key: value`` mapping between ``---`` fence
    lines at the very top of the file. Values may be wrapped in matching
    single or double quotes, which are stripped. Anything more complex
    (nested mappings, lists, multi-line values) raises
    :class:`SkillParseError` — guessing at YAML semantics with a regex is
    worse than refusing clearly.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillParseError(
            f"{path}: missing required frontmatter block "
            "(file must start with a '---' line)"
        )
    frontmatter: dict[str, str] = {}
    closing = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing = index
            break
        if not line.strip():
            continue
        match = _FRONTMATTER_KEY_RE.match(line)
        if match is None:
            raise SkillParseError(
                f"{path}: unsupported frontmatter line {line!r} "
                "(only flat 'key: value' entries are supported)"
            )
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        frontmatter[key] = value
    if closing is None:
        raise SkillParseError(f"{path}: frontmatter block is never closed ('---' missing)")
    body = "\n".join(lines[closing + 1 :])
    return frontmatter, body


@dataclass(frozen=True)
class Skill:
    """Metadata and lazy content accessor for one discovered skill.

    Instances are normally built by :meth:`from_directory` (which parses and
    validates the SKILL.md frontmatter) rather than directly. All path access
    goes through an internal :class:`~kinetic_sdk.workspace.manager.Workspace`
    rooted at :attr:`root_path`, so traversal payloads in resource paths are
    rejected by the same battle-tested logic the rest of the SDK uses —
    nothing here re-implements containment checks by hand.

    Attributes:
        name: Skill name; must match :data:`SKILL_NAME_PATTERN`, be at most
            :data:`MAX_NAME_LENGTH` chars, and equal the directory name.
        description: One-line summary shown to the model at discovery time.
            Must be non-empty — it is the *only* thing the model sees before
            deciding to load a skill. Truncated to
            :data:`MAX_DESCRIPTION_LENGTH` (with a notice appended) when
            longer.
        source: Where the skill came from: ``"local"``, ``"zip"``, or a
            future origin (e.g. a URL). Loaders set this; it is never read
            from user frontmatter, so a skill cannot lie about its provenance.
        root_path: Canonical absolute path of the skill directory.
        version: Optional free-form version string from frontmatter.
    """

    name: str
    description: str
    source: str
    root_path: str
    version: str | None = None

    def __post_init__(self) -> None:
        if len(self.name) > MAX_NAME_LENGTH or not SKILL_NAME_PATTERN.match(self.name):
            raise SkillParseError(
                f"invalid skill name {self.name!r}: must match "
                f"{SKILL_NAME_PATTERN.pattern!r} and be <= {MAX_NAME_LENGTH} chars"
            )
        if not self.description or not self.description.strip():
            raise SkillParseError(f"skill {self.name!r}: description must be non-empty")
        description = self.description
        if len(description) > MAX_DESCRIPTION_LENGTH:
            logger.warning(
                "skill %r description exceeds %d chars; truncating",
                self.name,
                MAX_DESCRIPTION_LENGTH,
            )
            notice = f"...[truncated, see full text at {self.source}]"
            description = description[: MAX_DESCRIPTION_LENGTH - len(notice)] + notice
            object.__setattr__(self, "description", description)
        object.__setattr__(self, "root_path", os.path.realpath(os.fspath(self.root_path)))

    # --- construction --------------------------------------------------

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str], source: str) -> "Skill":
        """Build a :class:`Skill` from a skill directory (frontmatter only).

        Reads SKILL.md, parses the frontmatter, and validates it. The
        Markdown *body* is discarded here — progressive disclosure keeps it
        unread until :meth:`read_main` is called explicitly.

        Args:
            directory: The skill directory. Its base name must equal the
                ``name`` declared in frontmatter (Agent Skills spec rule, not
                optional — a mismatch means the package was renamed or
                tampered with).
            source: Provenance label assigned by the loader (``"local"``,
                ``"zip"``, ...). Taken from the caller, never from the
                frontmatter, so a skill cannot forge a trusted origin.

        Raises:
            SkillParseError: SKILL.md missing, frontmatter malformed, a
                required field missing/invalid, or the frontmatter ``name``
                does not match the directory name.
        """
        directory = os.path.realpath(os.fspath(directory))
        skill_md = os.path.join(directory, SKILL_FILE_NAME)
        if not os.path.isfile(skill_md):
            raise SkillParseError(f"{directory}: missing required {SKILL_FILE_NAME}")
        text = Path(skill_md).read_text(encoding="utf-8")
        frontmatter, _body = _split_frontmatter(text, skill_md)
        name = frontmatter.get("name", "")
        dirname = os.path.basename(directory)
        if name != dirname:
            raise SkillParseError(
                f"{skill_md}: frontmatter name {name!r} does not match "
                f"directory name {dirname!r}"
            )
        return cls(
            name=name,
            description=frontmatter.get("description", ""),
            source=source,
            root_path=directory,
            version=frontmatter.get("version") or None,
        )

    # --- lazy content accessors -----------------------------------------

    def read_main(self) -> str:
        """Read the full SKILL.md body (everything after the frontmatter).

        Deliberately not cached: each call reads from disk so a removed
        directory surfaces immediately instead of serving a stale copy.

        Raises:
            SkillParseError: the SKILL.md is no longer readable — e.g. the
                root directory lived inside a
                :class:`~kinetic_sdk.skills.zip_loader.ZipSkillLoader` temp
                dir that has since been closed. The message names the skill
                and path so this never fails silently.
        """
        path = os.path.join(self.root_path, SKILL_FILE_NAME)
        if not os.path.isfile(path):
            raise SkillParseError(
                f"skill {self.name!r}: {SKILL_FILE_NAME} is not readable at {path!r} "
                "(the skill directory may have been removed — e.g. its "
                "ZipSkillLoader was closed)"
            )
        text = Path(path).read_text(encoding="utf-8")
        return _split_frontmatter(text, path)[1].strip()

    def list_resources(
        self,
        pattern: str | None = None,
        resource_type: ResourceType | None = None,
    ) -> list[str]:
        """List resource files as sorted root-relative POSIX paths.

        **The scope is hard-locked to the three sub-directories
        ``scripts/``, ``references/``, ``assets/``** — any other file in the
        skill root or in differently-named sub-directories is *not* listed,
        even when it physically exists after extraction. This is a deliberate
        difference from :class:`~kinetic_sdk.skills.zip_loader.ZipSkillLoader`'s
        ``allowed_extensions``: the zip allowlist controls which files may
        *exist on disk* after extraction (extraction-layer safety), while
        ``list_resources()`` controls which files the agent can *see* through
        the Skill API (exposure-layer safety). The two filters are
        independent and neither implies the other: a ``.txt`` sitting at the
        skill root may survive zip extraction (it matches the extension
        allowlist) yet never appears here — that is correct, not a bug.

        Args:
            pattern: Optional glob matched against each root-relative POSIX
                path (passed to
                :meth:`~kinetic_sdk.workspace.manager.Workspace.list_files`;
                ``*`` crosses directory separators).
            resource_type: Optionally restrict to one of ``"scripts"``,
                ``"references"``, ``"assets"`` — handy when a caller (e.g.
                the vetting scanner) only cares whether a skill ships
                executable scripts.

        Returns:
            Sorted root-relative POSIX paths under the resource directories.
        """
        workspace = Workspace(self.root_path)
        prefixes = (f"{resource_type}/",) if resource_type else tuple(
            f"{d}/" for d in RESOURCE_DIRS
        )
        return [
            path
            for path in workspace.list_files(pattern)
            if path.startswith(prefixes)
        ]

    def read_resource(self, relative_path: str) -> str:
        """Read one resource file by its skill-relative path.

        The path is resolved through
        :meth:`~kinetic_sdk.workspace.manager.Workspace.resolve` first, so
        ``../`` traversal and symlink escapes raise
        :class:`~kinetic_sdk.workspace.manager.PathTraversalError`. The
        resolved path must additionally live under one of the three resource
        directories — the exposure-layer scope of :meth:`list_resources`
        applies to reads too, otherwise listing and reading would disagree
        about what a "resource" is.

        Raises:
            PathTraversalError: the path escapes the skill root or points
                outside ``scripts/`` / ``references/`` / ``assets/``.
            FileNotFoundError: the resolved resource does not exist; the
                message names the skill and the requested path.
        """
        workspace = Workspace(self.root_path)
        resolved = workspace.resolve(relative_path)
        relative = PurePosixPath(os.path.relpath(resolved, self.root_path)).as_posix()
        if not any(relative.startswith(f"{d}/") for d in RESOURCE_DIRS):
            raise PathTraversalError(
                f"resource path {relative_path!r} is outside the resource "
                f"directories {RESOURCE_DIRS} of skill {self.name!r}"
            )
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"resource {relative_path!r} not found in skill {self.name!r}"
            )
        return Path(resolved).read_text(encoding="utf-8")
