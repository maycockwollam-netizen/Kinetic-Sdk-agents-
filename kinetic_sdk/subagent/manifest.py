"""The :class:`SubagentSpec`: declarative description of a spawnable sub-agent.

A spec is pure metadata — spawning itself lives in
:mod:`kinetic_sdk.subagent.delegation`. The two locked design rules this
module encodes:

1. **Every sub-agent has its own system prompt.** ``system_prompt`` is
   REQUIRED (never inherited from the parent) — this is the single
   deliberate difference from "an exact copy of the parent".
2. **Tools/permissions are NOT declared here.** Sub-agents inherit the
   parent's full tool set and ``permission_policy`` by default (the
   controller/executor pattern stays possible). A future version may add
   optional override fields to narrow that set; none are implemented now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from kinetic_sdk.skills.exceptions import SkillParseError
from kinetic_sdk.skills.skill import (
    MAX_NAME_LENGTH,
    SKILL_NAME_PATTERN,
    _split_frontmatter,
)
from kinetic_sdk.subagent.exceptions import SubagentSpecError

logger = logging.getLogger(__name__)

#: Sub-agent names follow the exact same shape as skill/plugin names
#: (lowercase alphanumeric words joined by single hyphens) — one naming
#: convention across every registry in the SDK.
SUBAGENT_NAME_PATTERN = SKILL_NAME_PATTERN

#: Maximum length of a sub-agent ``name`` (same bound as skills/plugins).
SUBAGENT_MAX_NAME_LENGTH = MAX_NAME_LENGTH


@dataclass(frozen=True)
class SubagentSpec:
    """Immutable description of a sub-agent a parent may delegate to.

    Attributes:
        name: Registered identifier used in ``DelegateTool`` calls. Must
            match :data:`SUBAGENT_NAME_PATTERN` and be at most
            :data:`SUBAGENT_MAX_NAME_LENGTH` chars.
        system_prompt: The sub-agent's OWN system message (required,
            non-empty). It never reuses the parent's system prompt.
        description: Short summary shown to the parent model so it can
            decide when delegating to this sub-agent is appropriate.
        model: Optional model override — a cheaper/faster model for the
            sub-agent. ``None`` (default) inherits the parent's model. A
            non-``None`` value only takes effect when the spawner is given
            an ``llm_factory`` able to build a client for it (see
            :func:`~kinetic_sdk.subagent.delegation.spawn_subagent`).
    """

    name: str
    system_prompt: str
    description: str = ""
    model: str | None = None

    def to_markdown(self) -> str:
        """Serialize this spec back to flat-frontmatter Markdown.

        Round-trips through :func:`load_subagent_spec`, so a profile can be
        written to a file, edited, and reloaded by name (the opencode
        ``.opencode/agent/*.md`` workflow). Empty fields are omitted.
        """
        lines = [f"name: {self.name}"]
        if self.description:
            lines.append(f"description: {self.description}")
        if self.model:
            lines.append(f"model: {self.model}")
        return "---\n" + "\n".join(lines) + "\n---\n\n" + self.system_prompt.rstrip() + "\n"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or (
            len(self.name) > SUBAGENT_MAX_NAME_LENGTH
            or not SUBAGENT_NAME_PATTERN.match(self.name)
        ):
            raise SubagentSpecError(
                f"Invalid sub-agent name {self.name!r}: must match "
                f"{SUBAGENT_NAME_PATTERN.pattern!r} and be <= "
                f"{SUBAGENT_MAX_NAME_LENGTH} chars"
            )
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise SubagentSpecError(
                f"Sub-agent {self.name!r} must declare a non-empty system_prompt "
                "(sub-agents never inherit the parent's system message)"
            )


def load_subagent_spec(path: str | Path) -> SubagentSpec:
    """Load a :class:`SubagentSpec` from flat-frontmatter Markdown.

    The parser is shared with skills so Markdown agent manifests have exactly
    the same deliberately small YAML-flat subset. Dataclass validation remains
    the single authority for required name and non-empty system prompt.
    """
    file_path = Path(path)
    try:
        frontmatter, body = _split_frontmatter(
            file_path.read_text(encoding="utf-8"), str(file_path)
        )
    except (OSError, SkillParseError) as exc:
        raise SubagentSpecError(str(exc)) from exc
    return SubagentSpec(
        name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        model=frontmatter.get("model") or None,
        system_prompt=body.strip(),
    )


def load_subagent_specs(agents_dir: str | Path) -> dict[str, SubagentSpec]:
    """Load every ``*.md`` in *agents_dir* as a :class:`SubagentSpec`.

    Files that fail to parse (bad frontmatter, invalid name, blank prompt)
    are skipped with a warning so one broken profile does not break the
    whole registry. Returns a mapping keyed by spec ``name``, ready to
    hand to :class:`~kinetic_sdk.subagent.tool.DelegateTool`.
    """
    directory = Path(agents_dir)
    if not directory.is_dir():
        raise SubagentSpecError(f"agents directory not found: {directory}")
    specs: dict[str, SubagentSpec] = {}
    for path in sorted(directory.glob("*.md")):
        try:
            spec = load_subagent_spec(path)
        except SubagentSpecError as exc:
            logger.warning("Skipping subagent file %s: %s", path, exc)
            continue
        specs[spec.name] = spec
    return specs
