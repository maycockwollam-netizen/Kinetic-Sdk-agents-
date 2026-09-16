"""Data objects used to record and inspect one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplayStep:
    """One chronologically ordered event or replay-valid state snapshot."""

    sequence: int
    timestamp: str
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayStep":
        sequence = data.get("sequence")
        timestamp = data.get("timestamp")
        event_type = data.get("event_type")
        payload = data.get("payload")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("step.sequence must be a non-negative int")
        if not isinstance(timestamp, str) or not isinstance(event_type, str):
            raise ValueError("step timestamp and event_type must be strings")
        if not isinstance(payload, dict):
            raise ValueError("step.payload must be a dict")
        return cls(sequence, timestamp, event_type, dict(payload))


@dataclass
class ReplayRun:
    """Persisted, append-only playback record for one ``Agent.run`` call."""

    run_id: str
    steps: list[ReplayStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayRun":
        run_id = data.get("run_id")
        steps = data.get("steps")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(steps, list):
            raise ValueError("steps must be a list")
        parsed = [ReplayStep.from_dict(item) for item in steps if isinstance(item, dict)]
        if len(parsed) != len(steps):
            raise ValueError("each step must be a dict")
        if [step.sequence for step in parsed] != list(range(len(parsed))):
            raise ValueError("step sequences must start at zero and be contiguous")
        return cls(run_id=run_id, steps=parsed)
