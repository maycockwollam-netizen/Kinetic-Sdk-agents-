"""OpenTelemetry export for the event bus.

The in-memory / console loggers are developer tools; production deployments
need the agent's event stream in the SAME tracing backend as the rest of
their infra (AutoGen exports OTel to Azure Monitor; the Claude SDK ships
optional OTel too). :class:`OTelObservabilityLogger` turns every agent
event into OTel spans/span-events WITHOUT changing the observable event
model — the bus wildcard still feeds everything here.

Semantics chosen (documented, do not silently change):

* ``agent.run_started`` OPENS a span named ``kinetic.run`` keyed by run id;
  every later event of that run becomes a span EVENT (attributes: event
  type + redacted payload fields, flattened to primitives) on it.
* ``agent.run_finished`` / ``agent.error`` / ``agent.cancelled`` CLOSE the
  run span (status ERROR for the failure/ cancel events, OK otherwise).
* Events without a run id (you get those from modules that publish before
  the run exists, like ``context.summarization_failed``) become standalone
  one-tick spans named after the event type.

``opentelemetry-sdk`` is optional (extra ``[otel]``). Like ``litellm``, the
import is lazy and only fails when you actually instantiate the logger —
importing this module is safe without the package installed.
"""

from __future__ import annotations

import logging
from typing import Any

from kinetic_sdk.event.bus import Event
from kinetic_sdk.observability.logger import ObservabilityLogger

logger = logging.getLogger(__name__)


class OTelObservabilityLogger(ObservabilityLogger):
    """Export agent events to OpenTelemetry as spans/span-events.

    Args:
        service_name: Resource/service name used when this logger creates
            its own tracer provider (ignored when one is passed).
        tracer_provider: Optional explicit provider; ``None`` uses the
            global one from ``otel.trace.get_tracer(__name__)`` — if no
            SDK is configured the spans are simply no-ops.

    The logger keeps run-id -> span bookkeeping internally; after the final
    run event the entry is dropped, so memory stays flat across runs.
    """

    def __init__(self, service_name: str = "kinetic-agent-sdk", tracer_provider: Any = None) -> None:
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OTelObservabilityLogger needs the 'opentelemetry-sdk' package "
                "(pip install 'kinetic-agent-sdk[otel]')"
            ) from exc
        self._trace = trace
        if tracer_provider is not None:
            provider = tracer_provider
        else:
            try:
                from opentelemetry.sdk.trace import (
                    TracerProvider,  # type: ignore[import-not-found]
                )
                provider = TracerProvider()
            except ImportError:
                provider = trace.get_tracer_provider()
        self._provider = provider
        self._tracer = self._trace.get_tracer(
            "kinetic-agent-sdk", tracer_provider=self._provider
        )
        self._open_spans: dict[str, Any] = {}
        self._service_name = service_name

    def handle(self, event: Event) -> None:
        """Turn one bus event into a span event (or run-span boundary)."""
        entry = self.build_entry(event)  # redaction stays centralised here
        payload = entry.get("payload") or {}
        run_id = entry.get("run_id") or payload.get("run_id")
        event_type = entry["event_type"]
        if event_type == "agent.run_started" and run_id:
            span = self._tracer.start_span("kinetic.run")
            span.set_attribute("kinetic.run_id", str(run_id))
            span.set_attribute("kinetic.service", self._service_name)
            self._open_spans[run_id] = span
            for key, value in self._flat(payload).items():
                span.set_attribute(str(key), value)
            return
        if run_id and run_id in self._open_spans:
            span = self._open_spans[run_id]
            attrs = self._flat(payload)
            span.add_event(event_type, attributes=attrs)
            if event_type in ("agent.run_finished", "agent.error", "agent.cancelled"):
                if event_type != "agent.run_finished":
                    span.set_status(self._status_error())
                span.end()
                del self._open_spans[run_id]
            return
        # No correlation: standalone event span.
        with self._tracer.start_as_current_span(event_type) as span:
            span.add_event(event_type, attributes=self._flat(payload))

    def flush_and_close(self) -> None:
        """End any open run spans and shut the provider down (idempotent)."""
        for span in list(self._open_spans.values()):
            span.end()
        self._open_spans.clear()
        shutdown = getattr(self._provider, "shutdown", None)
        if callable(shutdown):
            shutdown()

    # --- internals -----------------------------------------------------

    @staticmethod
    def _status_error() -> Any:
        from opentelemetry.trace.status import StatusCode  # type: ignore
        return StatusCode.ERROR

    @staticmethod
    def _flat(payload: dict[str, Any]) -> dict[str, Any]:
        """Flatten a redacted payload to OTel-safe primitives."""
        attrs: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (str, bool, int, float)):
                attrs[key] = value
            else:
                attrs[key] = str(value)
        return attrs
