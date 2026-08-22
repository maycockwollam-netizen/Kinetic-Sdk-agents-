"""Registry keeping hooks per :class:`HookPoint` and running them safely."""

from __future__ import annotations

import logging

from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.hooks.base import Hook, HookContext, HookPoint, HookResult
from kinetic_sdk.security.redact import redact_secrets

logger = logging.getLogger(__name__)


class HookRegistry:
    """Ordered collection of hooks, one list per :class:`HookPoint`.

    Args:
        event_bus: Optional bus used to publish ``hooks.error`` when a hook
            raises. When the registry is handed to an ``Agent`` without a bus
            of its own, the agent wires its own bus in so hook errors land in
            the same observability stream as the other agent events.

    Hooks registered for the same point run in registration order. A hook
    that raises is caught, logged, reported via ``hooks.error`` and skipped —
    it never crashes the agent loop and never blocks the remaining hooks.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus
        self._hooks: dict[HookPoint, list[Hook]] = {}

    def register(self, point: HookPoint, hook: Hook) -> None:
        """Register *hook* for *point*. Registering twice is a no-op."""
        hooks = self._hooks.setdefault(point, [])
        if hook not in hooks:
            hooks.append(hook)

    def unregister(self, point: HookPoint, hook: Hook) -> None:
        """Remove a previously registered hook. No-op if absent."""
        hooks = self._hooks.get(point)
        if hooks and hook in hooks:
            hooks.remove(hook)

    def hooks_for(self, point: HookPoint) -> list[Hook]:
        """The hooks registered for *point*, in registration order."""
        return list(self._hooks.get(point, []))

    def trigger(self, point: HookPoint, context: HookContext) -> list[HookResult]:
        """Run every hook registered for *point* and collect their results.

        Hooks returning ``None`` (pure observers) contribute no result.
        Exceptions are caught, logged and emitted as ``hooks.error``; the
        remaining hooks still run.
        """
        results: list[HookResult] = []
        for hook in self._hooks.get(point, []):
            try:
                result = hook(context)
            except Exception as exc:  # noqa: BLE001 - hooks must not break the loop
                logger.exception("Hook %r failed at %s", hook, point.value)
                self._emit_error(point, hook, exc)
                continue
            if result is not None:
                results.append(result)
        return results

    def _emit_error(self, point: HookPoint, hook: Hook, exc: Exception) -> None:
        """Publish ``hooks.error`` if a bus is attached (redacted message)."""
        if self.event_bus is None:
            return
        self.event_bus.publish(
            Event(
                type="hooks.error",
                payload={
                    "hook": getattr(hook, "__name__", repr(hook)),
                    "point": point.value,
                    "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                },
                source="hooks",
            )
        )
