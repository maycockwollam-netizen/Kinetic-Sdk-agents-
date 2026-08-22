# kinetic-agent-sdk

A modular agent SDK for the KINETIC coding agent. The SDK provides a
provider-neutral, synchronous tool-calling loop with full Python type hints and
clean interface/implementation separation so providers, tools, context
strategies, security policies, and observability sinks can be swapped without
rewriting the agent loop.

## Current status

- **Stage 1 — Core: done.** Includes the `Agent` loop, `Tool` / `ToolResult`,
  `ConversationState`, `EventBus`, the `LLMClient` interface, and the optional
  LiteLLM-backed client.
- **Stage 2 — Context & Memory: done for the current scope.** FLASH/MAX task
  routing, the model-backed `LiteLLMClassifier`, truncation-based context
  compaction, LLM-summarized context compaction with safe truncation fallback,
  and secret management are implemented.
- **Stage 3 — Quality & Ops: done.** Permission policies, audit logging,
  secret redaction, structured observability logging, per-run trace summaries,
  lifecycle hooks, public testing utilities, and hook-based confirmation UX
  are implemented. Richer policy sets, metrics, and external tracing exporters
  are still future work.
- **Stage 4 — Extensions: in progress.** Git integration (`GitTool`),
  workspace path safety (`Workspace`), agent profiles (dev/production
  presets), and full MCP support (client + server) are implemented. Subagents,
  plugins, and skills are not implemented yet.

## Module map

- `kinetic_sdk/agent/` — `Agent` tool-calling loop, `AgentMode` FLASH/MAX
  routing, task classification, mid-run FLASH -> MAX escalation.
- `kinetic_sdk/conversation/` — in-memory conversation history using an
  Anthropic-style typed message shape.
- `kinetic_sdk/context/` — context-window management with a zero-dependency
  token estimate and truncation-based compaction.
- `kinetic_sdk/event/` — synchronous/async event bus with wildcard subscribers.
- `kinetic_sdk/git/` — `GitTool`: curated git operations (status/diff/add/
  commit/branch/checkout/push/pull/log) as a first-class, policy-gated tool.
- `kinetic_sdk/hooks/` — lifecycle hooks (`BEFORE_RUN`, `BEFORE_TOOL_CALL`,
  `ON_PERMISSION_CHECK`, ...) with fail-safe semantics.
- `kinetic_sdk/llm/` — provider-neutral `LLMClient` interface plus optional
  `LiteLLMClient` for Anthropic/OpenAI-compatible providers.
- `kinetic_sdk/mcp/` — Model Context Protocol in both directions: client
  (connect to external MCP servers — Unity, filesystem, GitHub — and use
  their tools as Kinetic `Tool`s via stdio/SSE transports) and server
  (expose Kinetic tools to external MCP clients, with the same permission
  policy + audit log as the internal agent loop).
- `kinetic_sdk/observability/` — structured event loggers and `RunTrace`
  helpers for summarizing one agent run.
- `kinetic_sdk/profiles/` — ready-made agent configuration presets
  (`dev_profile`, `production_profile`).
- `kinetic_sdk/secret/` — secret value wrapper, providers, and registry for
  safe credential resolution.
- `kinetic_sdk/security/` — permission policies, audit loggers, and recursive
  secret redaction.
- `kinetic_sdk/testing/` — public test utilities: `MockLLMClient`, `MockTool`,
  and trace assertions.
- `kinetic_sdk/tool/` — abstract tool interface and `ToolResult` dataclass.
- `kinetic_sdk/workspace/` — `Workspace`: root-confined path resolution and
  file listing (path-traversal safe).

## Install (editable)

Install the core development dependencies:

```bash
pip install -e ".[dev]"
```

Install both development and LiteLLM backend dependencies when running the full
test suite or using `LiteLLMClient` / `LiteLLMClassifier`:

```bash
pip install -e ".[dev,llm]"
```

The core SDK intentionally has no required third-party runtime dependencies;
`litellm` is optional and imported lazily only when a LiteLLM-backed component
is instantiated.

## Run tests

Run the full suite after installing both extras:

```bash
python -m pytest -q
```

If only `.[dev]` is installed, tests that import or instantiate the LiteLLM
backend will fail because `litellm` is not present. Install `.[dev,llm]` before
using the LiteLLM backend tests.

## Minimal example

`LiteLLMClient` uses `litellm` under the hood, so one class reaches many
providers via the LiteLLM model string. The conversation history stays in
Anthropic message format (typed content blocks); the client translates
to/from the OpenAI shape that `litellm.completion` expects.

```python
from kinetic_sdk.agent import Agent
from kinetic_sdk.llm.client import LiteLLMClient
from kinetic_sdk.security.policy import AllowListPolicy
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


llm = LiteLLMClient(model="anthropic/claude-sonnet-4-5", api_key="sk-ant-...")

agent = Agent(
    llm=llm,
    tools=[EchoTool()],
    # The default policy is deny-by-default. Allow tools explicitly.
    permission_policy=AllowListPolicy(always_allow=["echo"]),
)
print(agent.run("Echo back: hello"))
```

For an OpenAI-compatible endpoint, provide both the LiteLLM model string and
`api_base`:

```python
llm = LiteLLMClient(
    model="openai/openhands/glm-5.2",
    api_key="...",
    api_base="https://llm-proxy.app.all-hands.dev",
)
```

## Security defaults

Tool execution is safe-by-default: when no `permission_policy` is provided,
`Agent` uses an empty `AllowListPolicy`, so every tool call is denied until the
SDK user explicitly allows that tool. Use `PermissivePolicy` only for local
development, sandboxed demos, or tests.

Tool inputs, outputs, audit entries, and observability payloads are redacted
before being persisted or published.

## Roadmap

- Add richer policy presets for filesystem, terminal, git, and network tools.
- Add metrics aggregation and external tracing exporters such as OpenTelemetry
  or Jaeger.
- Add the remaining Stage 4 extension points: subagents, plugins, and skills.
- MCP follow-ups: tool-list caching, server-initiated requests (sampling),
  and resources/prompts capabilities.
