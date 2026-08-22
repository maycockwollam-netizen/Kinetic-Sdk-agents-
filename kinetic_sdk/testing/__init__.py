"""Testing utilities: write deterministic agent tests without network calls.

This package ships the same style of fakes the SDK's own test suite uses, as
public utilities for SDK users: :class:`MockLLMClient` replays scripted LLM
turns, :class:`MockTool` stands in for real tools, and the assertion helpers
check a run's recorded :class:`~kinetic_sdk.observability.trace.RunTrace`.

Minimal end-to-end example — a complete agent test from scratch::

    from kinetic_sdk.agent.agent import Agent
    from kinetic_sdk.observability import InMemoryObservabilityLogger, RunTrace
    from kinetic_sdk.security import PermissivePolicy
    from kinetic_sdk.testing import (
        MockLLMClient,
        MockTool,
        assert_mode,
        assert_no_permission_denied,
        assert_tool_called,
        text_response,
        tool_response,
    )

    def test_my_agent():
        # 1. Script the model: first request a tool call, then answer.
        llm = MockLLMClient([
            tool_response("call-1", "calculator", {"expression": "1+1"}),
            text_response("The answer is 2."),
        ])

        # 2. Stand in for the real tool (no subprocess, no network).
        calc = MockTool(
            "calculator",
            description="Evaluate an arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            result="2",
        )

        # 3. Run the agent with observability on so the run can be traced.
        #    (PermissivePolicy: the default policy denies every tool.)
        obs = InMemoryObservabilityLogger()
        agent = Agent(
            llm=llm,
            tools=[calc],
            permission_policy=PermissivePolicy(),
            observability_logger=obs,
        )
        answer = agent.run("What is 1+1?")

        # 4. Assert on the recorded trace.
        trace = RunTrace.collect(obs.entries, agent.run_id)
        assert answer == "The answer is 2."
        assert calc.calls == [{"expression": "1+1"}]
        assert_tool_called(trace, "calculator", times=1)
        assert_mode(trace, "max")
        assert_no_permission_denied(trace)

Notes:
    * ``MockTool(result=ToolResult(error="boom"))`` (or a handler that
      raises) exercises error handling and FLASH -> MAX escalation.
    * ``MockLLMClient.calls`` / ``MockTool.calls`` record every invocation
      for direct assertions on what the agent sent.
"""

from kinetic_sdk.testing.assertions import (
    assert_mode,
    assert_no_permission_denied,
    assert_tool_called,
)
from kinetic_sdk.testing.mocks import (
    MockLLMClient,
    MockTool,
    text_response,
    tool_response,
)

__all__ = [
    "MockLLMClient",
    "MockTool",
    "assert_mode",
    "assert_no_permission_denied",
    "assert_tool_called",
    "text_response",
    "tool_response",
]
