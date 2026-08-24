"""Usage tracking: accumulator, agent wiring, trace aggregation."""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.llm.client import LLMResponse, ToolCall
from kinetic_sdk.llm.usage import UsageAccumulator
from kinetic_sdk.observability.logger import InMemoryObservabilityLogger
from kinetic_sdk.observability.trace import RunTrace
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response
from tests._helpers import EchoTool


def _used_response() -> LLMResponse:
    return LLMResponse(
        content="done",
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
    )


def test_accumulator_sums_and_ignores_garbage():
    acc = UsageAccumulator()
    acc.record({"input_tokens": 10, "output_tokens": 5})
    acc.record({"input_tokens": 3, "output_tokens": 2, "cost_usd": 0.002})
    acc.record({})  # no keys - still counts as a recorded call? no usage -> calls++
    snap = acc.snapshot()
    assert snap.input_tokens == 13
    assert snap.output_tokens == 7
    assert abs(snap.cost_usd - 0.002) < 1e-9
    # record({}) returns early: only 2 calls counted.
    assert snap.calls == 2


def test_snapshot_to_dict_round():
    acc = UsageAccumulator()
    acc.record({"input_tokens": 1, "output_tokens": 2, "cost_usd": 0.1})
    d = acc.snapshot().to_dict()
    assert d == {"input_tokens": 1, "output_tokens": 2, "cost_usd": 0.1, "calls": 1}


def test_agent_accumulates_across_runs():
    llm = MockLLMClient([_used_response(), _used_response()])
    agent = Agent(llm=llm, permission_policy=PermissivePolicy())
    agent.run("one")
    agent.run("two")
    snap = agent.usage.snapshot()
    assert snap.input_tokens == 20
    assert snap.output_tokens == 10
    assert snap.calls == 2


def test_agent_emits_llm_usage_events():
    llm = MockLLMClient([_used_response()])
    obs = InMemoryObservabilityLogger()
    agent = Agent(
        llm=llm, observability_logger=obs, permission_policy=PermissivePolicy()
    )
    agent.run("hi")
    usage_events = obs.get_events("llm.usage")
    assert len(usage_events) == 1
    assert usage_events[0]["payload"]["usage"]["input_tokens"] == 10
    # And run tracking still resolves the run id for the event.
    assert agent.run_id is not None
    trace = RunTrace.collect(obs.entries, agent.run_id)
    summary = trace.to_summary()
    assert summary["usage"]["input_tokens"] == 10
    assert summary["usage"]["output_tokens"] == 5
    assert summary["usage"]["calls"] == 1


def test_usage_accumulates_per_turn_with_tools():
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"message": "x"})],
                usage={"input_tokens": 7, "output_tokens": 3},
            ),
            LLMResponse(
                content="fin",
                stop_reason="end_turn",
                usage={"input_tokens": 11, "output_tokens": 4},
            ),
        ]
    )
    agent = Agent(
        llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy()
    )
    agent.run("echo x")
    snap = agent.usage.snapshot()
    assert snap.input_tokens == 18
    assert snap.calls == 2


def test_mock_response_without_usage_is_silence():
    llm = MockLLMClient([text_response("plain")])
    agent = Agent(llm=llm, permission_policy=PermissivePolicy())
    agent.run("hi")
    assert agent.usage.snapshot().calls == 0
