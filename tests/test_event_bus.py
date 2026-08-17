"""Unit tests for the EventBus."""

from __future__ import annotations

import asyncio

import pytest

from kinetic_sdk.event.bus import EventBus, Event


def test_sync_subscribe_and_publish():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("agent.run_started", seen.append)
    bus.publish(Event(type="agent.run_started", payload={"i": 1}))
    assert len(seen) == 1
    assert seen[0].type == "agent.run_started"
    assert seen[0].payload == {"i": 1}


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("x", seen.append)
    bus.unsubscribe("x", seen.append)
    bus.publish(Event(type="x"))
    assert seen == []


def test_wildcard_receives_all_events():
    bus = EventBus()
    all_events: list[Event] = []
    specific: list[Event] = []
    bus.subscribe("*", all_events.append)
    bus.subscribe("a.b", specific.append)
    bus.publish(Event(type="a.b", payload={"n": 1}))
    bus.publish(Event(type="c.d", payload={"n": 2}))
    assert [e.type for e in all_events] == ["a.b", "c.d"]
    assert [e.type for e in specific] == ["a.b"]


def test_duplicate_subscription_is_noop():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("x", seen.append)
    bus.subscribe("x", seen.append)  # second registration ignored
    bus.publish(Event(type="x"))
    assert len(seen) == 1


def test_subscriber_exception_does_not_break_publish():
    bus = EventBus()
    good: list[Event] = []

    def bad(_: Event) -> None:
        raise RuntimeError("listener exploded")

    bus.subscribe("x", bad)
    bus.subscribe("x", good.append)
    bus.publish(Event(type="x"))
    assert len(good) == 1


def test_async_subscriber_awaited():
    bus = EventBus()
    seen: list[Event] = []

    async def listener(event: Event) -> None:
        await asyncio.sleep(0)
        seen.append(event)

    bus.subscribe("x", listener)
    asyncio.run(bus.publish_async(Event(type="x", payload={"k": 1})))
    assert len(seen) == 1
    assert seen[0].payload == {"k": 1}


def test_async_mixed_sync_and_coroutine_subscribers():
    bus = EventBus()
    sync_seen: list[Event] = []
    async_seen: list[Event] = []

    async def coro(event: Event) -> None:
        async_seen.append(event)

    bus.subscribe("x", sync_seen.append)
    bus.subscribe("x", coro)
    asyncio.run(bus.publish_async(Event(type="x")))
    assert sync_seen and async_seen


def test_clear_removes_all_subscribers():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("x", seen.append)
    bus.subscribe("*", seen.append)
    bus.clear()
    bus.publish(Event(type="x"))
    assert seen == []


def test_event_source_is_preserved():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("x", seen.append)
    bus.publish(Event(type="x", payload={}, source="agent"))
    assert seen[0].source == "agent"
