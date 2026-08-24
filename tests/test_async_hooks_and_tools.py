"""Tests for the async extensions to hooks and tools.

``HookRegistry.trigger_async`` awaits coroutine hooks; the synchronous
``trigger`` rejects them with a warning instead of dropping them silently.
``Tool.execute_async`` bridges sync tools onto the event loop via a worker
thread, and ``AsyncMockTool`` is async-only by design.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.hooks.registry import HookRegistry
from kinetic_sdk.testing import AsyncMockTool, MockTool
from kinetic_sdk.tool.base import ToolResult

pytestmark = pytest.mark.asyncio


def make_context(point=HookPoint.BEFORE_RUN):
    return HookContext(point=point, run_id="test-run")


# --- HookRegistry.trigger_async ---------------------------------------------


async def test_trigger_async_awaits_coroutine_hooks():
    seen = []

    async def async_hook(ctx):
        await asyncio.sleep(0)
        seen.append("async")
        return HookResult(should_continue=True)

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, async_hook)
    results = await registry.trigger_async(HookPoint.BEFORE_RUN, make_context())
    assert seen == ["async"]
    assert len(results) == 1
    assert results[0].should_continue is True


async def test_trigger_async_runs_sync_hooks_too():
    seen = []
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, lambda ctx: seen.append("sync"))
    results = await registry.trigger_async(HookPoint.BEFORE_RUN, make_context())
    assert seen == ["sync"]
    assert results == []  # None-returning observers contribute no result


async def test_trigger_async_preserves_registration_order():
    seen = []

    async def first(ctx):
        seen.append("first")

    async def second(ctx):
        seen.append("second")

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, first)
    registry.register(HookPoint.BEFORE_RUN, second)
    await registry.trigger_async(HookPoint.BEFORE_RUN, make_context())
    assert seen == ["first", "second"]


async def test_trigger_async_raising_hook_does_not_block_others():
    from kinetic_sdk.event.bus import EventBus

    bus = EventBus()
    errors = []
    bus.subscribe("hooks.error", errors.append)
    seen = []

    async def bad(ctx):
        raise RuntimeError("async hook exploded")

    async def good(ctx):
        seen.append("good")

    registry = HookRegistry(event_bus=bus)
    registry.register(HookPoint.BEFORE_RUN, bad)
    registry.register(HookPoint.BEFORE_RUN, good)
    await registry.trigger_async(HookPoint.BEFORE_RUN, make_context())
    assert seen == ["good"]
    assert len(errors) == 1
    assert "async hook exploded" in errors[0].payload["error"]


# --- sync trigger rejects coroutine hooks ------------------------------------


@pytest.mark.asyncio
async def test_sync_trigger_rejects_coroutine_hooks(caplog):
    async def async_hook(ctx):
        return HookResult(should_continue=False)  # must NOT take effect

    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_RUN, async_hook)
    with caplog.at_level(logging.WARNING):
        results = registry.trigger(HookPoint.BEFORE_RUN, make_context())
    assert results == []  # rejected, not silently collected
    assert any("trigger_async" in r.message for r in caplog.records)


# --- Tool.execute_async default bridge ----------------------------------------


async def test_sync_tool_execute_async_bridge():
    tool = MockTool("calc", handler=lambda value: value * 2)
    result = await tool.execute_async(value=21)
    assert isinstance(result, ToolResult)
    assert result.output == 42
    assert tool.calls == [{"value": 21}]


async def test_execute_async_bridge_does_not_block_event_loop():
    tool = MockTool("slow", handler=lambda: time.sleep(0.1) or "done")
    ticked = asyncio.Event()

    async def ticker():
        await asyncio.sleep(0.02)
        ticked.set()

    helper = asyncio.create_task(ticker())
    result = await tool.execute_async()
    await helper
    assert result.output == "done"
    assert ticked.is_set()


async def test_execute_async_bridge_propagates_exceptions():
    def boom():
        raise RuntimeError("sync tool exploded")

    tool = MockTool("boom", handler=boom)
    with pytest.raises(RuntimeError, match="sync tool exploded"):
        await tool.execute_async()


# --- AsyncMockTool ------------------------------------------------------------


async def test_async_mock_tool_fixed_result():
    tool = AsyncMockTool("echo", result="fixed")
    result = await tool.execute_async(message="hi")
    assert result.output == "fixed"
    assert tool.calls == [{"message": "hi"}]


async def test_async_mock_tool_async_handler():
    async def handler(**params):
        await asyncio.sleep(0)
        return ToolResult(output=f"async:{params['message']}")

    tool = AsyncMockTool("echo", handler=handler)
    result = await tool.execute_async(message="yo")
    assert result.output == "async:yo"


async def test_async_mock_tool_sync_handler_also_works():
    tool = AsyncMockTool("echo", handler=lambda message: f"sync:{message}")
    result = await tool.execute_async(message="yo")
    assert result.output == "sync:yo"


async def test_async_mock_tool_sync_execute_raises():
    tool = AsyncMockTool("echo", result="x")
    with pytest.raises(NotImplementedError, match="async-only"):
        tool.execute(message="hi")


@pytest.mark.asyncio
async def test_async_mock_tool_rejects_result_and_handler_together():
    with pytest.raises(ValueError, match="either a fixed result or a handler"):
        AsyncMockTool("echo", result="x", handler=lambda: "y")
