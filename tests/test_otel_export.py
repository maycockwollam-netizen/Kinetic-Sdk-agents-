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


def test_otel_tool_becomes_nested_span_with_duration(otel_setup):
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
    tool_span = next(
        s for s in exporter.get_finished_spans()
        if s.name == "kinetic.tool" and s.kind is not None
    )
    assert tool_span.parent.span_id == root.context.span_id
    assert tool_span.attributes.get("kinetic.tool") == "echo"
    assert isinstance(tool_span.attributes.get("kinetic.duration_seconds"), float)


def test_otel_llm_response_becomes_nested_span_with_usage(otel_setup):
    provider, exporter = otel_setup
    bus = EventBus()
    logger = OTelObservabilityLogger(tracer_provider=provider)
    logger.attach(bus)
    llm = MockLLMClient([text_response("done")])
    agent = Agent(
        llm=llm,
        event_bus=bus,
        permission_policy=PermissivePolicy(),
        run_budget=None,
    )
    agent.run("hi")
    logger.flush_and_close()
    run_span = next(s for s in exporter.get_finished_spans() if s.name == "kinetic.run")
    llm_span = next(s for s in exporter.get_finished_spans() if s.name == "kinetic.llm")
    assert llm_span.parent.span_id == run_span.context.span_id
    assert llm_span.attributes.get("kinetic.stop_reason") in ("end_turn", "stop", "done")
    # Tool calls were promoted to nested spans, not left as run-span events.
    run_event_names = [e.name for e in run_span.events]
    assert "agent.tool_call_finished" not in run_event_names


def test_otel_subagent_run_nests_under_parent_via_context(otel_setup):
    provider, exporter = otel_setup
    bus = EventBus()
    logger = OTelObservabilityLogger(tracer_provider=provider)
    logger.attach(bus)
    from kinetic_sdk.subagent import DelegateTool, SpawnBudget, SubagentSpec

    def parent_llm(messages, tools, system):  # type: ignore[no-untyped-def]
        from kinetic_sdk.llm.client import LLMResponse, ToolCall

        return LLMResponse(
            tool_calls=[ToolCall(id="d1", name="delegate", arguments={"subagent_name": "child", "task_prompt": "work"})],
            stop_reason="tool_use",
        )

    child_llm = MockLLMClient([text_response("child done")])
    parent = Agent(
        MockLLMClient([parent_llm]),
        event_bus=bus,
        permission_policy=PermissivePolicy(),
    )

    delegate = DelegateTool(
        [SubagentSpec(name="child", system_prompt="you are child", description="child")],
        budget=SpawnBudget(100),
        llm_factory=lambda model: child_llm,
    )
    parent.add_tool(delegate)
    delegate.bind(parent)
    parent.run("delegate please")
    logger.flush_and_close()

    run_spans = [s for s in exporter.get_finished_spans() if s.name == "kinetic.run"]
    assert len(run_spans) == 2  # parent + child
    parent_span = next(s for s in run_spans if s.attributes.get("kinetic.run_id") == parent.run_id)
    child_spans = [s for s in run_spans if s is not parent_span]
    assert len(child_spans) == 1
    assert child_spans[0].parent.span_id == parent_span.context.span_id
