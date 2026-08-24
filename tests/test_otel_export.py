"""OpenTelemetry export: run spans, events, and lifecycle."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response
from tests._helpers import EchoTool

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore
    InMemorySpanExporter,
)

from kinetic_sdk.observability.otel import OTelObservabilityLogger


@pytest.fixture()
def otel_setup():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_otel_exports_run_span_with_events(otel_setup):
    provider, exporter = otel_setup
    bus = EventBus()
    logger = OTelObservabilityLogger(tracer_provider=provider)
    logger.attach(bus)
    llm = MockLLMClient([text_response("done")])
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    agent.run("hi")
    logger.flush_and_close()
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "kinetic.run")
    event_names = [e.name for e in root.events]
    assert "agent.turn_started" in event_names
    assert "agent.run_finished" in event_names
    assert root.attributes["kinetic.run_id"] == agent.run_id


def test_otel_uncorrelated_events_become_spans(otel_setup):
    provider, exporter = otel_setup
    bus = EventBus()
    logger = OTelObservabilityLogger(tracer_provider=provider)
    logger.attach(bus)
    from kinetic_sdk.event.bus import Event

    bus.publish(
        Event(type="context.summarization_failed", payload={"reason": "x"})
    )
    logger.flush_and_close()
    spans = exporter.get_finished_spans()
    assert any(s.name == "context.summarization_failed" for s in spans)


def test_otel_finished_run_dropped_from_bookkeeping(otel_setup):
    provider, exporter = otel_setup
    logger = OTelObservabilityLogger(tracer_provider=provider)
    bus = EventBus()
    logger.attach(bus)
    llm = MockLLMClient([text_response("a"), text_response("b")])
    agent = Agent(llm=llm, event_bus=bus, permission_policy=PermissivePolicy())
    agent.run("one")
    agent.run("two")
    logger.flush_and_close()
    assert logger._open_spans == {}
    assert len([s for s in exporter.get_finished_spans() if s.name == "kinetic.run"]) == 2


def test_otel_tool_event_flattened_into_span(otel_setup):
    provider, exporter = otel_setup
    bus = EventBus()
    logger = OTelObservabilityLogger(tracer_provider=provider)
    logger.attach(bus)
    from kinetic_sdk.testing import tool_response

    llm = MockLLMClient(
        [
            tool_response("c1", "echo", {"message": "x"}),
            text_response("done"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        event_bus=bus,
        permission_policy=PermissivePolicy(),
    )
    agent.run("echo x")
    logger.flush_and_close()
    root = next(s for s in exporter.get_finished_spans() if s.name == "kinetic.run")
    names = [e.name for e in root.events]
    assert "agent.tool_call_finished" in names
