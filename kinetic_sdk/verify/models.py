"""Verification contract for coding agents.

Month-2 roadmap item: an agent must not claim it finished work without
evidence. This module defines the shape of that evidence so it can be
attached to ``Agent.run`` results, stored in eval artifacts, or fed back
for repair when the proof is weak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerificationEvidence:
    """Proof that an agent run actually touched files and ran checks."""

    changed_files: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    test_results: dict[str, str] = field(default_factory=dict)
    remaining_risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_verified(self) -> bool:
        """A run is only "verified" with files touched, commands run and a
        successful test result — otherwise it must not be reported complete."""
        passing = {"passed", "pass", "ok", "success", "green"}
        return bool(
            self.changed_files
            and self.commands_run
            and any(str(v).lower() in passing for v in self.test_results.values())
        )


def verification_schema() -> dict[str, Any]:
    """JSON-Schema subset for structured output validation."""
    return {
        "type": "object",
        "properties": {
            "changed_files": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "test_results": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "remaining_risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["changed_files", "commands_run", "test_results"],
        "additionalProperties": False,
    }
