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
- **Stage 4 — Extensions: done.** Git integration (`GitTool`), workspace
  path safety (`Workspace`), agent profiles (dev/production presets), full
  MCP support (client + server), skills (discovery + vetting), plugins
  (dynamic loading with static scanning), and subagents (delegation with
  inherited permissions plus budget/circuit-breaker guardrails) are all
  implemented.
- **Hardening pass — done.** Runtime resilience on top of the four stages:
  LLM retry/backoff + timeout, per-tool wall-clock timeout, parallel-group-safe
  context compaction (no orphaned `tool_result`, oversized tool outputs are
  trimmed head/tail), pluggable token counters (`tiktoken` extra), sticky-MAX
  routing across runs, conversation persistence (`JsonFileConversationStore`),
  cooperative cancellation (`agent.cancel()`), tool-input schema validation,
  optional parallel tool execution, and the standard `terminal` +
  `file_editor` tools.
- **Stage 5 — Production readiness: done.** Structured output
  (`run(output_schema=...)`), usage/cost tracking, OpenTelemetry export
  (`otel` extra), metrics aggregation, ask-user human-in-the-loop tool,
  long-term memory (keyword-ranked, JSON-file durable), a stdlib REST
  `AgentServer`, checkpoint fork/rewind helpers, an eval harness, docker
  sandbox adapters for `TerminalTool`, async sub-agent delegation, and MCP
  reconnect. See `CHANGELOG.md` for the full detail.
- **Async layer — done.** `AsyncAgent` mirrors every sync capability
  (routing, compaction, hooks, policy, structured output, memory, usage).

## Module map

- `kinetic_sdk/agent/` — `Agent` tool-calling loop, `AgentMode` FLASH/MAX
  routing, task classification, mid-run FLASH -> MAX escalation.
- `kinetic_sdk/conversation/` — conversation history using an Anthropic-style
  typed message shape, plus pluggable persistence (`ConversationStore` ABC,
  `JsonFileConversationStore`) for save/resume across processes.
- `kinetic_sdk/context/` — context-window management: zero-dependency token
  estimate or pluggable real tokenizers (`TiktokenCounter`), parallel-safe
  truncation compaction, oversized tool-result trimming, and optional
  LLM-summarised compaction with safe fallback.
- `kinetic_sdk/event/` — synchronous/async event bus with wildcard subscribers.
- `kinetic_sdk/files/` — `FileTool`: workspace-confined view/create/
  str_replace/insert/undo_edit (all paths traversal-safe via `Workspace`).
- `kinetic_sdk/terminal/` — `TerminalTool`: shell commands with process-group
  kill on timeout and head/tail output truncation.
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
- `kinetic_sdk/plugin/` — dynamic loading of external Python packages that
  register extra tools: metadata-only discovery (entry points + PLUGIN.md
  directory convention), an AST-based static scanner (a tripwire, NOT a
  sandbox — see the package docstring), scan-before-import loading with full
  audit logging, and collision-rejecting tool merge.
- `kinetic_sdk/profiles/` — ready-made agent configuration presets
  (`dev_profile`, `production_profile`).
- `kinetic_sdk/secret/` — secret value wrapper, providers, and registry for
  safe credential resolution.
- `kinetic_sdk/security/` — permission policies, audit loggers, and recursive
  secret redaction.
- `kinetic_sdk/skills/` — Markdown skill packages: metadata-first discovery,
  zip loading with extraction guards, and static + LLM-assisted vetting for
  untrusted sources.
- `kinetic_sdk/subagent/` — sub-agent delegation: `DelegateTool` spawns
  sub-agents that inherit the parent's full tool set and permission policy
  (each with its own system prompt and fresh context), guarded by a shared
  tree-wide tool-call budget and per-agent repetition circuit breakers.
- `kinetic_sdk/testing/` — public test utilities: `MockLLMClient`, `MockTool`,
  and trace assertions.
- `kinetic_sdk/ask_user/` — `AskUserTool`: the agent asks the operator a
  question mid-run through an injectable handler (human-in-the-loop).
- `kinetic_sdk/eval/` — eval harness: cases, callable scorers, and an
  `EvalRunner` with per-case `RunTrace` reporting.
- `kinetic_sdk/memory/` — long-term memory providers (in-memory + JSON
  file), `MemoryTool`, and the `Agent(memory=...)` auto recall/store wiring.
- `kinetic_sdk/server/` — `AgentServer`: stdlib-only REST API for one-shot
  agent runs plus optional server-managed workspace endpoints (bearer-token
  auth optional).
- `kinetic_sdk/tool/` — abstract tool interface and `ToolResult` dataclass.
- `kinetic_sdk/workspace/` — `WorkspaceBase` with local, Docker and remote
  backends; `Workspace` remains the root-confined local implementation.

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

Add the `tokens` extra for a real tokenizer (tiktoken) backing context-window
estimation instead of the built-in heuristic:

```bash
pip install -e ".[tokens]"
```

The core SDK intentionally has no required third-party runtime dependencies;
`litellm` and `tiktoken` are optional and imported lazily only when a
component backed by them is instantiated.

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
- Add external tracing exporters beyond the included OpenTelemetry support,
  such as Jaeger.
- Subagent follow-ups: parallel sub-agent execution (the shared budget is
  already thread-safe), optional tool-narrowing overrides on `SubagentSpec`.
- MCP follow-ups: tool-list caching, server-initiated requests (sampling),
  and resources/prompts capabilities.
- Plugin follow-ups: `hook` capability, unload/hot-reload, and (research
  only — never a promise) real isolation such as subprocess-hosted plugins.
