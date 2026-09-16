"""In-memory cursor API for inspecting a recorded replay step by step."""

from __future__ import annotations

from kinetic_sdk.replay.models import ReplayRun, ReplayStep
from kinetic_sdk.replay.store import ReplayStore, ReplayStoreError


class ReplayDebugger:
    """Navigate a durable replay without re-calling an LLM or executing tools."""

    def __init__(self, replay: ReplayRun) -> None:
        self.replay = replay
        self._position = -1

    @classmethod
    def open(cls, store: ReplayStore) -> "ReplayDebugger":
        replay = store.load()
        if replay is None:
            raise ReplayStoreError("No replay has been saved")
        return cls(replay)

    @property
    def position(self) -> int:
        """Zero-based cursor position; ``-1`` means before the first step."""
        return self._position

    def steps(self) -> list[ReplayStep]:
        """Return the immutable replay steps in chronological order."""
        return list(self.replay.steps)

    def current(self) -> ReplayStep | None:
        """Return the selected step, or ``None`` before/after the timeline."""
        if 0 <= self._position < len(self.replay.steps):
            return self.replay.steps[self._position]
        return None

    def seek(self, sequence: int) -> ReplayStep | None:
        """Move to *sequence*; ``len(steps)`` represents the end position."""
        if not isinstance(sequence, int) or not -1 <= sequence <= len(self.replay.steps):
            raise IndexError("Replay sequence is outside the recorded timeline")
        self._position = sequence
        return self.current()

    def next(self) -> ReplayStep | None:
        """Advance one step, returning ``None`` after the final step."""
        return self.seek(min(self._position + 1, len(self.replay.steps)))

    def previous(self) -> ReplayStep | None:
        """Move back one step, returning ``None`` before the first step."""
        return self.seek(max(self._position - 1, -1))
