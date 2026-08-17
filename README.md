# Kinetic-Sdk-agents-

# kinetic-agent-sdk

A modular agent SDK for the KINETIC coding agent. Stage 1 provides the core
tool-calling loop; later stages add context management, security, observability
and multi-agent orchestration. The SDK is written in Python with full type
hints and a clean interface/implementation separation so providers and
strategies can be swapped without touching the agent loop.

## Stage 1 — Core (current)

- `kinetic_sdk/agent/` — `Agent` tool-calling loop, `AgentMode` (FLASH/MAX),
  `TaskClassifier` interface (stub).
- `kinetic_sdk/conversation/` — `ConversationState` in-memory history.
- `kinetic_sdk/event/` — `EventBus` publish/subscribe.
- `kinetic_sdk/llm/` — `LLMClient` interface + `LiteLLMClient` (optional, multi-provider).
- `kinetic_sdk/tool/` — abstract `Tool` / `ToolResult`.
- `kinetic_sdk/context/` — `ContextManager` interface (Stage 2 stub).

## Install (editable)

```bash
pip install -e ".[dev]"
# For the LLM backend (Anthropic Claude, OpenAI-compatible endpoints, ...):
pip install -e ".[llm]"
```

## Run tests

```bash
pytest -q
```

## Minimal example

`LiteLLMClient` uses `litellm` under the hood, so one class reaches many
providers via the LiteLLM model string. The conversation history stays in
Anthropic message format (typed content blocks); the client translates
to/from the OpenAI shape that `litellm.completion` expects.

```python
from kinetic_sdk.agent import Agent
from kinetic_sdk.conversation import ConversationState
from kinetic_sdk.event import EventBus
from kinetic_sdk.llm.client import LiteLLMClient
from kinetic_sdk.tool.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echo a message."
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, message: str) -> ToolResult:
        return ToolResult(output=message)


# Anthropic Claude via the "anthropic/" provider prefix:
llm = LiteLLMClient(model="anthropic/claude-sonnet-4-5", api_key="sk-ant-...")

# Or an OpenAI-compatible endpoint (e.g. the OpenHands proxy) via api_base:
# llm = LiteLLMClient(
#     model="openai/openhands/glm-5.2",
#     api_key="...",
#     api_base="https://llm-proxy.app.all-hands.dev",
# )

agent = Agent(llm=llm, tools=[EchoTool()])
print(agent.run("Echo back: hello"))
```

## Roadmap

- Stage 2 — context window management + secret handling.
- Stage 3 — security, observability, hooks, testing utils, critic.
- Stage 4 — subagents, plugins, MCP, skills, workspace, git, profiles.