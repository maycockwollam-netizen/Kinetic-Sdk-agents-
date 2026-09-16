"""Recorded tool doubles for deterministic continuation of a replay branch."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from kinetic_sdk.conversation.store import state_from_dict
from kinetic_sdk.replay.models import ReplayRun
from kinetic_sdk.tool.base import Tool, ToolResult


class DeterministicReplayError(Exception):
    """A recorded tool result cannot safely satisfy a replayed call."""


@dataclass(frozen=True)
class RecordedToolCall:
    """One tool invocation and its exact model-visible result."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult


class ReplayTool(Tool):
    """A tool that returns recorded results in call order and never has side effects."""

    description = "Deterministic result recorded from an earlier agent run."
    parameters: dict[str, Any] = {"type": "object", "additionalProperties": True}

    def __init__(self, name: str, calls: list[RecordedToolCall], *, strict_inputs: bool) -> None:
        self.name = name
        self._calls = deque(calls)
        self._strict_inputs = strict_inputs

    def execute(self, **params: Any) -> ToolResult:
        if not self._calls:
            return ToolResult(error=f"No recorded result remains for tool {self.name!r}")
        recorded = self._calls.popleft()
        if self._strict_inputs and recorded.arguments != params:
            return ToolResult(
                error=(
                    f"Recorded input mismatch for tool {self.name!r}: "
                    f"expected {recorded.arguments!r}, got {params!r}"
                )
            )
        return copy.deepcopy(recorded.result)


class DeterministicToolReplay:
    """Extract recorded tool results from raw replay snapshots.

    Pass :meth:`tools` to a new branch agent in place of real side-effecting
    tools.  This makes a changed-model continuation repeat the old tool
    outcomes.  It requires snapshots captured with
    ``ReplayRecorder(capture_raw_snapshots=True)``; redacted snapshots are
    intentionally rejected rather than replaying corrupted data.
    """

    def __init__(self, calls: list[RecordedToolCall], *, strict_inputs: bool = True) -> None:
        self.calls = list(calls)
        self.strict_inputs = strict_inputs

    @classmethod
    def from_replay(
        cls, replay: ReplayRun, *, strict_inputs: bool = True
    ) -> "DeterministicToolReplay":
        call_defs: dict[str, tuple[str, dict[str, Any]]] = {}
        results: dict[str, ToolResult] = {}
        for step in replay.steps:
            if step.event_type != "replay.snapshot":
                continue
            state_data = step.payload.get("state")
            if not isinstance(state_data, dict):
                continue
            _ensure_unredacted(state_data)
            state = state_from_dict(state_data)
            for message in state.messages:
                content = message.get("content")
                if message.get("role") == "assistant" and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            call_id, name, arguments = (
                                block.get("id"),
                                block.get("name"),
                                block.get("input"),
                            )
                            if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, dict):
                                call_defs[call_id] = (name, dict(arguments))
                if message.get("role") == "user" and isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            continue
                        call_id = block.get("tool_use_id")
                        if isinstance(call_id, str) and call_id not in results:
                            results[call_id] = _tool_result_from_block(block)
        calls = [
            RecordedToolCall(name, arguments, results[call_id])
            for call_id, (name, arguments) in call_defs.items()
            if call_id in results
        ]
        return cls(calls, strict_inputs=strict_inputs)

    def tools(self) -> list[ReplayTool]:
        """Return one deterministic tool per recorded tool name."""
        grouped: dict[str, list[RecordedToolCall]] = defaultdict(list)
        for call in self.calls:
            grouped[call.name].append(call)
        return [
            ReplayTool(name, calls, strict_inputs=self.strict_inputs)
            for name, calls in grouped.items()
        ]


def _tool_result_from_block(block: dict[str, Any]) -> ToolResult:
    content = block.get("content")
    if block.get("is_error"):
        try:
            parsed = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            return ToolResult(error=parsed["error"])
        return ToolResult(error=str(content))
    return ToolResult(output=content)


def _ensure_unredacted(value: Any) -> None:
    if isinstance(value, str):
        if "[REDACTED]" in value:
            raise DeterministicReplayError(
                "Replay contains redacted snapshots; record with "
                "capture_raw_snapshots=True in protected storage to replay tools"
            )
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_unredacted(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_unredacted(item)
