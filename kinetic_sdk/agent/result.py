"""A richer, first-class result object for an agent run.

``Agent.run()`` returns the final assistant ``str`` (kept for backward
compatibility), but a caller often wants more: the parsed structured output,
how many tokens it cost, the trace id, the mode it ran in, and whether it was
cancelled. :class:`AgentResult` bundles all of it. The agent fills
``agent.last_result`` after every completed run (accessible even when you only
ask for the ``str``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kinetic_sdk.agent.modes import AgentMode


@dataclass(frozen=True)
class AgentResult:
    """Outcome of ONE ``Agent.run``, alongside the final text.

    Attributes:
        final_text: The same string ``run()`` returns.
        structured_output: The parsed value when the run used an
            ``output_schema`` (``None`` otherwise / when it failed).
        run_id: Stable id of this run (matches observability events).
        mode: The agent mode the run executed in (FLASH/MAX).
        usage: Cumulative token/cost snapshot for the run (see
            :meth:`~kinetic_sdk.llm.usage.UsageSnapshot.to_dict`).
        duration_seconds: Wall-clock length of the run.
        cancelled: ``True`` when the run was stopped by :meth:`Agent.cancel`.
        error: Failure detail when the run ended with an error.
    """

    final_text: str
    structured_output: Any = None
    run_id: str | None = None
    mode: AgentMode | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None
    cancelled: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "structured_output": self.structured_output,
            "run_id": self.run_id,
            "mode": self.mode.value if self.mode is not None else None,
            "usage": self.usage,
            "duration_seconds": self.duration_seconds,
            "cancelled": self.cancelled,
            "error": self.error,
        }