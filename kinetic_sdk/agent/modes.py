"""Agent execution modes.

The KINETIC agent runs in two modes:

* :attr:`AgentMode.FLASH` - fast path for simple tasks: minimal tool calls, no
  planner/verifier. Lower latency and cost.
* :attr:`AgentMode.MAX` - full planner -> executor -> verifier pipeline with a
  larger context budget. Used for complex tasks.

Routing is decided by :class:`kinetic_sdk.agent.classifier.TaskClassifier`
(Stage 2 wiring). Stage 1 defines the enum and helpers so the agent can carry
the concept without the routing logic yet. Escalation FLASH -> MAX mid-task is
allowed via :meth:`AgentMode.escalates_to`; the reverse is intentionally not
supported within a single task.
"""

from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    """Execution mode of the agent.

    Subclassing ``str`` so the value serialises naturally to JSON/text and
    compares equal to its string value (``AgentMode.FLASH == "flash"``).
    """

    FLASH = "flash"
    MAX = "max"

    @classmethod
    def escalates_to(cls, current: "AgentMode") -> "AgentMode | None":
        """Return the mode to escalate to, or ``None`` if not allowed.

        Only FLASH -> MAX escalation is permitted within a single task;
        downgrading MAX -> FLASH returns ``None``.
        """
        if current == cls.FLASH:
            return cls.MAX
        return None
