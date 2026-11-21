"""Per-turn reasoning-effort adjustment for the agent loop.

Inspired by opencode's ``effort-updates``: a model that supports dynamic
reasoning effort (e.g. OpenAI's ``reasoning_effort``) can run routine turns
cheaper while keeping hard turns thorough. The policy is a plain callable so
users can roll their own; the shipped :class:`AdaptiveEffortPolicy` starts
strong, keeps tool-calling turns on medium effort and drops to medium for
idle (non-tool) turns.

Providers that don't support per-request effort get no parameter at all, so
enabling a policy never breaks a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Effort(str, Enum):
    """Reasoning-effort level; the value is the provider-facing string."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class EffortContext:
    """Input a policy uses to pick effort for the upcoming LLM call.

    Attributes:
        iteration: Zero-based turn index inside the run.
        previous_had_tool_calls: Whether the previous turn requested tool
            calls (the current turn is therefore a continuation).
    """

    iteration: int
    previous_had_tool_calls: bool


class EffortPolicy(Protocol):
    """Plug-in point: given the loop context, return the effort to use.

    Return ``None`` to leave the provider's default effort untouched.
    """

    def effort_for(self, context: EffortContext) -> Effort | None: ...


@dataclass(frozen=True)
class AdaptiveEffortPolicy:
    """Simple default: start strong, stay medium on continuations, rest medium.

    Attributes:
        initial: Effort for the first turn of a run (default ``high``).
        tool_turn: Effort after a turn that called tools (default ``medium``).
        idle_turn: Effort after a turn that did not call tools (default
            ``medium``) - the run is close to finishing, so a lighter turn
            saves cost.
    """

    initial: Effort = Effort.HIGH
    tool_turn: Effort = Effort.MEDIUM
    idle_turn: Effort = Effort.MEDIUM

    def effort_for(self, context: EffortContext) -> Effort:
        if context.iteration == 0:
            return self.initial
        if context.previous_had_tool_calls:
            return self.tool_turn
        return self.idle_turn