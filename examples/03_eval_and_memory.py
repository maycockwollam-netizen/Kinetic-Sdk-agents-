"""Eval harness + long-term memory (offline demo)."""

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.eval import EvalCase, EvalRunner, exact_match, no_tool_failures
from kinetic_sdk.memory import InMemoryMemory, MemoryTool
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.testing import MockLLMClient, text_response

memory = InMemoryMemory()


def factory() -> Agent:
    return Agent(
        llm=MockLLMClient([text_response("4")]),
        tools=[MemoryTool(memory)],
        permission_policy=AllowListPolicy(always_allow=["memory"]),
        memory=memory,
    )


cases = [
    EvalCase("what is 2+2?", expected="4", tags=["math"]),
    EvalCase("what is 2+2?", expected="4", tags=["math"]),  # second run recalls the first
]

report = EvalRunner(factory, scorers=[exact_match, no_tool_failures]).run(cases)
print("pass rate:", report["pass_rate"])
print("memory entries after eval:", len(memory.all()))
