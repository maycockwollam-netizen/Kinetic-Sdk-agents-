"""Deterministic, opt-in caching of tool results.

Agents often re-ask the same READ-ONLY question (a file view, a grep, a git
status) and pay the tool + model cost again for an identical answer.
:class:`ToolCache` + :class:`CachedTool` memoize the result of identical
(``tool``, ``args``) calls within a short TTL, so repeated lookups are served
from memory.

Safety by design:

* Only ``CachedTool`` (an opt-in wrapper) caches — the agent loop never does
  this automatically, because a WRITE-bearing tool (terminal, file editor,
  git) can change state between calls and stale results would be wrong.
* Only **non-error** results are cached; a failure is retried next time.
* The key is the tool name + the exact JSON-serialisable arguments (sorted),
  so different arguments never collide.
* A short TTL (default 2s) bounds how long a result is reused.

This is a building block for a future per-run cache; it is deliberately NOT
wired into ``Agent`` core.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

from kinetic_sdk.tool.base import Tool, ToolResult


def tool_cache_key(name: str, arguments: dict[str, Any]) -> str:
    """Stable cache key: tool name plus canonicalised arguments."""
    payload = json.dumps(
        {"name": name, "arguments": arguments}, sort_keys=True, default=str, separators=(",", ":")
    )
    return payload


class ToolCache:
    """Thread-safe LRU cache of ``ToolResult`` keyed by tool+arguments.

    Args:
        maxsize: Maximum number of cached results (LRU eviction).
        ttl_seconds: How long a cached result stays valid. ``None`` keeps
            results until evicted.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 2.0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._entries: "OrderedDict[str, tuple[float, ToolResult]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ToolResult | None:
        """Return the cached result for *key* if still fresh, else None."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, result = entry
            if self._ttl is not None and time.monotonic() - stored_at > self._ttl:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return result

    def put(self, key: str, result: ToolResult) -> None:
        """Store *result* for *key*, evicting the LRU entry when full.

        Error results are NOT stored (a failed call should be re-attempted).
        """
        if result.is_error:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), result)
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class CachedTool(Tool):
    """Opt-in wrapper adding a :class:`ToolCache` in front of another tool.

    A cached, non-error result for the exact same arguments is returned
    without re-running the inner tool. Only use this for READ-ONLY,
    deterministic tools (grep, glob, view, git status) — see the module
    docstring for the safety rationale.
    """

    def __init__(
        self,
        inner: Tool,
        cache: ToolCache | None = None,
        ttl_seconds: float = 2.0,
    ) -> None:
        if not isinstance(inner, Tool):
            raise TypeError("inner must be a Tool")
        self._inner = inner
        self._cache = cache if cache is not None else ToolCache(ttl_seconds=ttl_seconds)
        # Mirror the wrapped tool's identity so the agent treats this as one
        # and the same tool.
        self.name = inner.name
        self.description = inner.description
        self.parameters = inner.parameters
        self.thread_safe = getattr(inner, "thread_safe", False)

    def execute(self, **params: Any) -> ToolResult:  # type: ignore[override]
        """Serve from cache or run the inner tool and cache the result."""
        key = tool_cache_key(self.name, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._inner.execute(**params)
        self._cache.put(key, result)
        return result