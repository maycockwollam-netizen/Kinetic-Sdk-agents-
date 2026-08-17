"""Lightweight publish/subscribe event bus.

Modules communicate through events instead of direct imports. This keeps the
core decoupled: the agent loop emits events ("tool_call_started",
"llm_response_received", ...) and any subscriber (logger, UI, metrics) can
react without the emitter knowing about it.

Design goals for Stage 1:
* Synchronous dispatch (simple, deterministic, easy to test).
* Subscribers receive an :class:`Event` with a string ``type`` and a free-form
  ``payload`` dict. Type safety is intentionally loose here; stricter typed
  events can be layered on later without changing the bus contract.
* Subscribers may be regular functions or coroutines. Coroutine subscribers
  are awaited when :meth:`EventBus.publish` is awaited via
  :meth:`EventBus.publish_async`; sync subscribers are simply called in both
  modes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger(__name__)

#: A subscriber is either a sync function (event -> None) or a coroutine
#: function (event -> Awaitable[None]).
Subscriber = Union[Callable[["Event"], None], Callable[["Event"], Awaitable[None]]]


@dataclass
class Event:
    """An event flowing through the bus.

    Attributes:
        type: Dotted string identifier, e.g. ``"agent.loop_started"``.
        payload: Arbitrary, JSON-serialisable data attached to the event.
        source: Optional name of the emitting component, useful for routing
            or debugging.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


class EventBus:
    """A minimal in-memory pub/sub bus.

    The bus is not thread-safe; it is meant to be used within a single
    asyncio event loop / thread. Cross-process distribution is a Stage 4
    concern.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = {}

    def subscribe(self, event_type: str, subscriber: Subscriber) -> None:
        """Register *subscriber* to be called for events of *event_type*.

        The same subscriber can be registered for multiple event types.
        Registering the same callable twice for the same type is a no-op.
        ``"*"`` is a wildcard type that receives every published event.
        """
        self._subscribers.setdefault(event_type, [])
        if subscriber not in self._subscribers[event_type]:
            self._subscribers[event_type].append(subscriber)

    def unsubscribe(self, event_type: str, subscriber: Subscriber) -> None:
        """Remove a previously registered subscriber. No-op if absent."""
        subs = self._subscribers.get(event_type)
        if subs and subscriber in subs:
            subs.remove(subscriber)

    def _gather(self, event_type: str) -> list[Subscriber]:
        """Return subscribers for a type plus the wildcard subscribers."""
        return [*self._subscribers.get("*", []), *self._subscribers.get(event_type, [])]

    def publish(self, event: Event) -> None:
        """Dispatch *event* to all matching sync subscribers.

        Coroutine subscribers are skipped here; use :meth:`publish_async` to
        also await them. Exceptions raised by a subscriber are logged and
        swallowed so one bad listener cannot break the emitter.
        """
        for sub in self._gather(event.type):
            if inspect.iscoroutinefunction(sub):
                continue
            try:
                sub(event)
            except Exception:  # noqa: BLE001 - listeners must not break emitter
                logger.exception("Sync subscriber %r failed for %s", sub, event.type)

    async def publish_async(self, event: Event) -> None:
        """Dispatch *event*, awaiting coroutine subscribers along the way.

        Sync subscribers are called directly. Each coroutine subscriber is
        awaited sequentially (not concurrently) to keep ordering simple and
        predictable for Stage 1.
        """
        for sub in self._gather(event.type):
            try:
                result = sub(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - listeners must not break emitter
                logger.exception("Subscriber %r failed for %s", sub, event.type)

    def clear(self) -> None:
        """Remove all subscribers. Mainly useful in tests."""
        self._subscribers.clear()
