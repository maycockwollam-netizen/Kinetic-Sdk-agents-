"""Quickstart: a runnable agent in 30 lines (offline — MockLLMClient).

Swap MockLLMClient for LiteLLMClient(model="...") to hit a real provider.
"""

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.memory import InMemoryMemory, MemoryTool
from kinetic_sdk.observability import ConsoleObservabilityLogger, RunTrace
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.terminal import TerminalTool
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from kinetic_sdk.observability.logger import InMemoryObservabilityLogger

# 1) A scripted LLM so the example runs offline: turn 1 runs a shell
#    command, turn 2 answers with the result.
llm = MockLLMClient(
    [
        tool_response("c1", "terminal", {"command": "echo hello from the sandbox"}),
        text_response("The command printed: hello from the sandbox"),
    ]
)

# 2) Tools + policy (deny-by-default AllowListPolicy, opt-in per tool).
tools = [TerminalTool(), MemoryTool(InMemoryMemory())]
policy = AllowListPolicy(always_allow=["terminal", "memory"])

# 3) Observability: in-memory entries we turn into a trace afterwards.
obs = InMemoryObservabilityLogger()
agent = Agent(llm=llm, tools=tools, permission_policy=policy, observability_logger=obs)

answer = agent.run("Say hello from the sandbox")
print("FINAL:", answer)
print("SUMMARY:", RunTrace.collect(obs.entries, agent.run_id).to_summary())
