"""Sliding-window detection for repeated tool calls."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any


class StuckDetector:
    """Observe recent tool calls and detect repeated name/argument pairs."""

    def __init__(self, window_size: int = 6, repeat_threshold: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if repeat_threshold < 1:
            raise ValueError("repeat_threshold must be positive")
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self._calls: deque[tuple[str, str]] = deque(maxlen=window_size)

    @property
    def window(self) -> list[dict[str, str]]:
        """A serialisable snapshot of the most recent observed calls."""
        return [
            {"tool_name": tool_name, "args_hash": args_hash}
            for tool_name, args_hash in self._calls
        ]

    def observe(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Return whether this call appears at least threshold times in the window."""
        serialized = json.dumps(tool_args, sort_keys=True, separators=(",", ":"), default=str)
        args_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        call = (tool_name, args_hash)
        self._calls.append(call)
        return sum(candidate == call for candidate in self._calls) >= self.repeat_threshold
