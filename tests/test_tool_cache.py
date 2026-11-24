"""Tests for tool/cache.py — ToolCache + CachedTool."""

from __future__ import annotations

import time

from kinetic_sdk.tool import CachedTool, Tool, ToolCache, ToolResult


class CountingTool(Tool):
    name = "count"
    description = "x"
    parameters: dict = {"type": "object", "properties": {"n": {"type": "integer"}}}
    thread_safe = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, n: int = 0, **_: object) -> ToolResult:
        self.calls += 1
        return ToolResult(output=f"c{n}")


def test_cached_tool_serves_repeat_call_from_cache() -> None:
    inner = CountingTool()
    tool = CachedTool(inner)
    a = tool.execute(n=1)
    b = tool.execute(n=1)
    assert a.output == b.output == "c1"
    assert inner.calls == 1  # second identical call was cached


def test_different_arguments_are_not_cached() -> None:
    inner = CountingTool()
    tool = CachedTool(inner)
    tool.execute(n=1)
    tool.execute(n=2)
    assert inner.calls == 2


def test_error_results_are_not_cached() -> None:
    class Flaky(Tool):
        name = "flaky"
        description = "x"
        parameters: dict = {"type": "object", "properties": {}}
        thread_safe = True

        def execute(self, **_: object) -> ToolResult:
            return ToolResult(error="boom")

    inner = Flaky()
    tool = CachedTool(inner)
    assert tool.execute().is_error
    assert tool.execute().is_error  # re-ran (not cached)
    assert len(tool._cache) == 0


def test_ttl_expiry_reruns() -> None:
    inner = CountingTool()
    tool = CachedTool(inner, ttl_seconds=0.05)
    tool.execute(n=9)
    assert inner.calls == 1
    time.sleep(0.07)
    tool.execute(n=9)
    assert inner.calls == 2


def test_tool_cache_lru_eviction() -> None:
    cache = ToolCache(maxsize=2, ttl_seconds=None)
    cache.put("a", ToolResult(output="A"))
    cache.put("b", ToolResult(output="B"))
    cache.put("c", ToolResult(output="C"))
    assert len(cache) == 2
    # "a" was evicted, so it is a miss now.
    assert cache.get("a") is None
    assert cache.get("b").output == "B"


def test_cache_key_is_stable_regardless_of_arg_order() -> None:
    from kinetic_sdk.tool import tool_cache_key

    assert tool_cache_key("t", {"a": 1, "b": 2}) == tool_cache_key("t", {"b": 2, "a": 1})