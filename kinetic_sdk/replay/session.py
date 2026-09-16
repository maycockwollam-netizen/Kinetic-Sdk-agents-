"""Forking, comparison and timeline view models for replay debugging."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, cast

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import state_from_dict
from kinetic_sdk.replay.deterministic import DeterministicToolReplay
from kinetic_sdk.replay.models import ReplayRun, ReplayStep

if TYPE_CHECKING:
    from kinetic_sdk.llm.client import LLMClient
    from kinetic_sdk.tool.base import Tool


class ReplayForkError(Exception):
    """A selected replay position cannot be used as a continuation branch."""


@dataclass(frozen=True)
class TimelineEntry:
    """Compact, UI-ready representation of an event in a debug timeline."""

    sequence: int
    timestamp: str
    event_type: str
    label: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReplayDiffEntry:
    """One aligned or changed point in two replay timelines."""

    kind: str
    left: ReplayStep | None
    right: ReplayStep | None


@dataclass(frozen=True)
class ReplayDiff:
    """Structured comparison suitable for a timeline/debug-session UI."""

    left_run_id: str
    right_run_id: str
    entries: list[ReplayDiffEntry]

    @property
    def changed(self) -> list[ReplayDiffEntry]:
        return [entry for entry in self.entries if entry.kind != "equal"]

    @property
    def is_equal(self) -> bool:
        return not self.changed


class ReplayBranch:
    """A mutable continuation prepared from a replay-valid snapshot."""

    def __init__(self, parent_run_id: str, fork_sequence: int, state: ConversationState) -> None:
        self.parent_run_id = parent_run_id
        self.fork_sequence = fork_sequence
        self.state = state

    def replace_input(self, text: str) -> "ReplayBranch":
        """Replace the latest plain user turn, or append one when none exists."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input text must be a non-empty string")
        for message in reversed(self.state.messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["content"] = text
                return self
        self.state.add_user_message(text)
        return self

    def add_input(self, text: str) -> "ReplayBranch":
        """Append a new user turn before continuing the branch."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input text must be a non-empty string")
        self.state.add_user_message(text)
        return self

    def create_agent(
        self, llm: "LLMClient", tools: Iterable["Tool"] | None = None, **agent_kwargs: Any
    ) -> "Any":
        """Build a sync Agent from this point, allowing model/tool replacement."""
        from kinetic_sdk.agent.agent import Agent

        if "state" in agent_kwargs:
            raise ValueError("ReplayBranch owns state; do not pass state=")
        return Agent(llm=llm, tools=tools, state=copy.deepcopy(self.state), **agent_kwargs)

    def create_recorder(self, store: "Any", *, capture_raw_snapshots: bool = False) -> "Any":
        """Create a recorder that persists this branch's parent lineage."""
        from kinetic_sdk.replay.recorder import ReplayRecorder

        return ReplayRecorder(
            store,
            capture_raw_snapshots=capture_raw_snapshots,
            parent_run_id=self.parent_run_id,
            fork_sequence=self.fork_sequence,
        )

    def deterministic_tools(self, replay: ReplayRun, *, strict_inputs: bool = True) -> list["Tool"]:
        """Return no-side-effect tools that repeat results captured in *replay*."""
        return cast(
            list["Tool"],
            DeterministicToolReplay.from_replay(replay, strict_inputs=strict_inputs).tools(),
        )


class ReplayDebugSession:
    """High-level debug API: inspect, fork, compare and render a timeline."""

    def __init__(self, replay: ReplayRun) -> None:
        self.replay = replay

    def timeline(self) -> list[TimelineEntry]:
        return [
            TimelineEntry(
                sequence=step.sequence,
                timestamp=step.timestamp,
                event_type=step.event_type,
                label=_label(step),
                payload=copy.deepcopy(step.payload),
            )
            for step in self.replay.steps
        ]

    def fork(self, sequence: int | None = None) -> ReplayBranch:
        """Fork from the latest replay-valid snapshot at or before *sequence*."""
        target = len(self.replay.steps) - 1 if sequence is None else sequence
        if not isinstance(target, int) or target < 0 or target >= len(self.replay.steps):
            raise ReplayForkError("fork sequence is outside the recorded timeline")
        snapshots = [
            step for step in self.replay.steps[: target + 1] if step.event_type == "replay.snapshot"
        ]
        if not snapshots:
            raise ReplayForkError("no replay-valid snapshot exists at or before this sequence")
        snapshot = snapshots[-1]
        state_data = snapshot.payload.get("state")
        if not isinstance(state_data, dict):
            raise ReplayForkError("snapshot has no conversation state")
        if _contains_redaction(state_data):
            raise ReplayForkError(
                "snapshot is redacted; use protected raw snapshots to fork a live conversation"
            )
        try:
            state = state_from_dict(copy.deepcopy(state_data))
        except (TypeError, ValueError, KeyError) as exc:
            raise ReplayForkError(f"snapshot state is invalid: {exc}") from exc
        return ReplayBranch(self.replay.run_id, snapshot.sequence, state)

    def compare(self, other: ReplayRun) -> ReplayDiff:
        """Align timelines by stable content, deliberately ignoring timestamps/run IDs."""
        entries: list[ReplayDiffEntry] = []
        max_len = max(len(self.replay.steps), len(other.steps))
        for index in range(max_len):
            left = self.replay.steps[index] if index < len(self.replay.steps) else None
            right = other.steps[index] if index < len(other.steps) else None
            if left is None:
                kind = "added"
            elif right is None:
                kind = "removed"
            elif _step_key(left) == _step_key(right):
                kind = "equal"
            else:
                kind = "changed"
            entries.append(ReplayDiffEntry(kind, left, right))
        return ReplayDiff(self.replay.run_id, other.run_id, entries)

    def timeline_json(self) -> str:
        """Return a transport-friendly timeline payload for a web/CLI UI."""
        return json.dumps(
            [
                {
                    "sequence": entry.sequence,
                    "timestamp": entry.timestamp,
                    "event_type": entry.event_type,
                    "label": entry.label,
                    "payload": entry.payload,
                }
                for entry in self.timeline()
            ],
            ensure_ascii=False,
            default=str,
        )


_IDENTITY_FIELDS = frozenset({"run_id"})


def _stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return payload content without per-run identifiers at any depth."""
    return {
        key: _stable_value(value)
        for key, value in payload.items()
        if key not in _IDENTITY_FIELDS
    }


def _stable_value(value: Any) -> Any:
    """Recursively remove identity fields without mutating recorded payloads."""
    if isinstance(value, dict):
        return _stable_payload(value)
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_value(item) for item in value)
    return value


def _step_key(step: ReplayStep) -> str:
    return json.dumps(
        {"event_type": step.event_type, "payload": _stable_payload(step.payload)},
        sort_keys=True,
        default=str,
    )


def _label(step: ReplayStep) -> str:
    payload = step.payload
    if step.event_type == "agent.tool_call_started":
        return f"Tool started: {payload.get('name', 'unknown')}"
    if step.event_type == "agent.tool_call_finished":
        status = "failed" if payload.get("is_error") else "finished"
        return f"Tool {status}: {payload.get('name', 'unknown')}"
    if step.event_type == "replay.snapshot":
        return "Replay-valid checkpoint"
    return step.event_type.replace(".", " · ")


def _contains_redaction(value: Any) -> bool:
    if isinstance(value, str):
        return "[REDACTED]" in value
    if isinstance(value, dict):
        return any(_contains_redaction(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_redaction(item) for item in value)
    return False
