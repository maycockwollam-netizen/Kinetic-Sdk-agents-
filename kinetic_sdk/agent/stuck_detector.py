"""Sliding-window detection for repeated tool calls."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from kinetic_sdk.tool.base import ToolResult


class StuckDetector:
    """Detect repeated calls whose inputs *and results* make no progress."""

    def __init__(self, window_size: int = 6, repeat_threshold: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if repeat_threshold < 1:
            raise ValueError("repeat_threshold must be positive")
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self._calls: deque[tuple[str, str, str]] = deque(maxlen=window_size)

    @property
    def window(self) -> list[dict[str, str]]:
        """A serialisable snapshot of the most recent observed calls."""
        return [
            {"tool_name": tool_name, "args_hash": args_hash, "result_hash": result_hash}
            for tool_name, args_hash, result_hash in self._calls
        ]

    def observe(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Backward-compatible input-only observation API.

        New agent-loop code should use :meth:`check`, which additionally
        requires an unchanged result before declaring an unproductive loop.
        """
        return self._observe(tool_name, tool_args, {"legacy": True})

    def check(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> bool:
        """Return ``True`` for a repeated call with an identical outcome."""
        outcome = {
            "output": result.output,
            "error": result.error,
            "failure_category": (
                result.failure_category.value if result.failure_category is not None else None
            ),
        }
        return self._observe(tool_name, arguments, outcome)

    def _observe(self, tool_name: str, tool_args: dict[str, Any], outcome: Any) -> bool:
        serialized = json.dumps(tool_args, sort_keys=True, separators=(",", ":"), default=str)
        outcome_serialized = json.dumps(
            outcome, sort_keys=True, separators=(",", ":"), default=str
        )
        args_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        result_hash = hashlib.sha256(outcome_serialized.encode("utf-8")).hexdigest()
        call = (tool_name, args_hash, result_hash)
        self._calls.append(call)
        return sum(candidate == call for candidate in self._calls) >= self.repeat_threshold
