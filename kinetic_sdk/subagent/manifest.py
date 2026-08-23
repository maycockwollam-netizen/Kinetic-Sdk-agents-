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

from dataclasses import dataclass

from kinetic_sdk.skills.skill import MAX_NAME_LENGTH, SKILL_NAME_PATTERN
from kinetic_sdk.subagent.exceptions import SubagentSpecError

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
