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
            from opentelemetry import (
                context,  # type: ignore[import-not-found]
                trace,  # type: ignore[import-not-found]
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OTelObservabilityLogger needs the 'opentelemetry-sdk' package "
                "(pip install 'kinetic-agent-sdk[otel]')"
            ) from exc
        self._context = context
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
        #: run_id -> the open ``kinetic.run`` span.
        self._open_spans: dict[str, Any] = {}
        #: run_id -> attach token making that run's span the current OTel
        #: context (so a sub-agent run spawned in this thread nests under it).
        self._attach_tokens: dict[str, Any] = {}
        #: (run_id, tool call id) -> the open ``kinetic.tool`` span.
        self._tool_spans: dict[str, dict[str, Any]] = {}
        #: run_id -> the most recent ``llm.usage`` payload (merged into the
        #: ``kinetic.llm`` span that follows it).
        self._last_usage: dict[str, dict[str, Any]] = {}
        self._service_name = service_name

    def handle(self, event: Event) -> None:
        """Turn one bus event into spans / span-events.

        Lifecycle: ``agent.run_started`` opens the ``kinetic.run`` span and
        makes it the current OTel context (a sub-agent run in this thread
        nests under it). LLM calls become one-shot ``kinetic.llm`` spans
        (cost/tokens from the preceding ``llm.usage``), tool calls become
        ``kinetic.tool`` spans spanning start->finish with duration/error,
        and remaining events stay span events on the run span. Events without
        a run id become standalone one-tick spans.
        """
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
            # Make this run's span the current OTel context: a sub-agent run
            # started synchronously in this thread inherits it as its parent.
            self._attach_tokens[run_id] = self._context.attach(
                self._trace.set_span_in_context(span)
            )
            return
        if run_id and run_id in self._open_spans:
            span = self._open_spans[run_id]
            parent_context = self._trace.set_span_in_context(span)

            if event_type == "agent.tool_call_started":
                tool_span = self._tracer.start_span("kinetic.tool", context=parent_context)
                tool_span.set_attribute("kinetic.run_id", str(run_id))
                tool_span.set_attribute("kinetic.tool", str(payload.get("name")))
                tool_span.set_attribute("kinetic.tool_call_id", str(payload.get("id")))
                self._tool_spans.setdefault(run_id, {})[str(payload.get("id"))] = tool_span
                return
            if event_type == "agent.tool_call_finished":
                tool_span = self._tool_spans.get(run_id, {}).pop(str(payload.get("id")), None)
                if tool_span is not None:
                    tool_span.set_attribute("kinetic.tool", str(payload.get("name")))
                    if payload.get("is_error"):
                        tool_span.set_status(self._status_error())
                    duration = payload.get("duration_seconds")
                    if isinstance(duration, (int, float)):
                        tool_span.set_attribute("kinetic.duration_seconds", float(duration))
                    tool_span.end()
                return
            if event_type == "llm.usage":
                usage = payload.get("usage")
                self._last_usage[run_id] = dict(usage) if isinstance(usage, dict) else {}
                return
            if event_type == "agent.llm_response":
                llm_span = self._tracer.start_span("kinetic.llm", context=parent_context)
                llm_span.set_attribute("kinetic.run_id", str(run_id))
                llm_span.set_attribute("kinetic.tool_calls", int(payload.get("tool_calls") or 0))
                llm_span.set_attribute("kinetic.stop_reason", str(payload.get("stop_reason")))
                usage = self._last_usage.pop(run_id, {})
                for key in ("input_tokens", "output_tokens"):
                    if isinstance(usage.get(key), int):
                        llm_span.set_attribute(f"kinetic.{key}", usage[key])
                cost = usage.get("cost_usd")
                if isinstance(cost, (int, float)):
                    llm_span.set_attribute("kinetic.cost_usd", float(cost))
                llm_span.end()
                return

            span.add_event(event_type, attributes=self._flat(payload))
            if event_type in ("agent.run_finished", "agent.error", "agent.cancelled"):
                if event_type != "agent.run_finished":
                    span.set_status(self._status_error())
                span.end()
                token = self._attach_tokens.pop(run_id, None)
                if token is not None:
                    self._context.detach(token)
                self._tool_spans.pop(run_id, None)
                self._last_usage.pop(run_id, None)
                del self._open_spans[run_id]
            return
        # No correlation: standalone event span.
        with self._tracer.start_as_current_span(event_type) as standalone:
            standalone.add_event(event_type, attributes=self._flat(payload))

    def flush_and_close(self) -> None:
        """End any open spans and shut the provider down (idempotent)."""
        for run_id, span in list(self._open_spans.items()):
            span.end()
            token = self._attach_tokens.pop(run_id, None)
            if token is not None:
                self._context.detach(token)
        for run_id in list(self._tool_spans):
            for tool_span in self._tool_spans[run_id].values():
                tool_span.end()
        self._open_spans.clear()
        self._attach_tokens.clear()
        self._tool_spans.clear()
        self._last_usage.clear()
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
