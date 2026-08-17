# AGENTS.md — kinetic-agent-sdk

Repository-specific knowledge for OpenHands agents working on this SDK.

## Overview
`kinetic-agent-sdk` is a self-authored agent SDK that will replace Claude Agent
SDK inside the KINETIC coding agent. Architecture is inspired by OpenHands
`software-agent-sdk` but reimplemented from scratch, no copied code.

## Build / Test commands
- Install (dev): `pip install -e ".[dev]"`
- Install (llm backend, optional): `pip install -e ".[llm]"` (pulls in `litellm`)
- Run tests: `python -m pytest -q` (85 tests: 53 Stage 1 + 12 LiteLLMClient +
  5 modes/classifier + 20 classifier+routing/escalation)
- No build step beyond pip install.

## Stage status
- Stage 1 (Core): DONE — `tool`, `event`, `llm`, `conversation`, `agent` loop.
- Stage 2 (Context & Memory): PARTIAL — `agent/classifier.py` real impl +
  FLASH/MAX routing + escalation DONE (see "Classifier & routing" below).
  `context/manager.py` still an interface stub (`NoopContextManager`); needs
  token-aware truncation/summarisation. `secret/` module not yet created.
- Stage 3 (Quality & Ops): TODO.
- Stage 4 (Extensions): TODO.

## Key design rules
- All communication in Vietnamese during task work (per user instruction).
- Python 3.10+ type hints everywhere. Interfaces (ABC) separate from impls.
- `litellm` is an OPTIONAL dependency; never hard-import it at module top
  level — `LiteLLMClient` imports it lazily in `_import_litellm` (called
  from `__init__`). Core stays zero-dependency (stdlib only).
- No third-party deps for Stage 1 core. `ToolResult`/`Event` use dataclasses,
  NOT pydantic.
- Agent loop is synchronous + deterministic for Stage 1. Async variants
  (`AsyncLLMClient`) are interface-only for now.
- FLASH/MAX modes: classifier uses a SEPARATE cheap model, hidden behind
  alias `kinetic-classifier-v1` (never leak real provider/model name).
  Escalation FLASH->MAX allowed mid-task; MAX->FLASH is NOT within one task.

## Gotchas learned
- `ConversationState` defines `__len__`, so an empty state is falsy. Always
  use `is not None` (NOT `or`) when defaulting constructor args of this type
  in the `Agent` — see `agent/agent.py`.
- MockLLM replays a scripted list of `LLMResponse`/callables for deterministic
  agent tests (no network). Helper lives in `tests/_helpers.py`.

## Module map (Stage 1 + Stage 2 classifier)
- `kinetic_sdk/agent/agent.py` — `Agent.run()` tool-calling loop, emits events;
  classifies once before the first turn, routes FLASH/MAX, escalates mid-run.
- `kinetic_sdk/agent/modes.py` — `AgentMode` enum + `escalates_to`.
- `kinetic_sdk/agent/classifier.py` — `TaskComplexity` enum, `Classification`
  dataclass, `TaskClassifier` ABC, `DefaultClassifier` (stub, always MAX),
  `LiteLLMClassifier` (real, model-backed, alias `kinetic-classifier-v1`).
- `kinetic_sdk/conversation/state.py` — `ConversationState` (Anthropic message shape).
- `kinetic_sdk/event/bus.py` — `EventBus` (sync + async dispatch, wildcard `*`).
- `kinetic_sdk/llm/client.py` — `LLMClient` ABC, `LiteLLMClient` (litellm-backed,
  Anthropic<->OpenAI translation), `LLMResponse`, `ToolCall`, `StreamEvent`,
  `AsyncLLMClient` ABC.
- `kinetic_sdk/tool/base.py` — `Tool` ABC + `ToolResult` dataclass.
- `kinetic_sdk/context/manager.py` — `ContextManager` ABC + `NoopContextManager` stub.

## Classifier & routing (Stage 2 — DONE)
- `LiteLLMClassifier` drives `openai/openhands/glm-5.2` via
  `api_base=https://llm-proxy.app.all-hands.dev`, reading `api_key` from the
  `OPENHANDS_API_KEY` env var (NEVER hardcoded). The real model/api_base are
  private constants (`_MODEL`/`_API_BASE`); only the alias
  `kinetic-classifier-v1` ever surfaces in logs, events, `Classification`.
  rationale, or `LiteLLMClassifier.model`. Prompt asks for ONE word
  (SIMPLE/COMPLEX), `max_tokens=10`. Any failure (exception / unparseable
  reply) falls back to `TaskComplexity.COMPLEX` (safe side -> MAX).
- `Agent.run` classifies exactly ONCE before the first turn (never re-invoked
  on escalation). Result sets `self.mode` + `max_iterations`
  (`MODE_MAX_ITERATIONS`: FLASH=5, MAX=50) + `enable_extended_reasoning`
  (False/True). Emits `agent.classified`.
- Mid-run escalation FLASH->MAX when (a) the first turn's tool call(s) error,
  or (b) `FLASH_ESCALATION_THRESHOLD` (3) iterations pass with no final answer.
  Escalation emits `agent.escalated` once, raises the cap to the MAX default,
  keeps conversation state. MAX->FLASH is NOT allowed. An explicit user
  `max_iterations` override is respected for the initial mode; escalation still
  raises the cap to the MAX default (unless the override is pinned).
- `max_iterations` constructor default changed from `25` to `None` (routing
  picks per mode). Existing tests that passed an explicit value are unaffected.

## Next task (LLM backend switched to LiteLLM — DONE)
Build `agent/classifier.py` real implementation + wire `AgentMode` routing into
`Agent.run` (currently `mode` defaults to MAX, classifier not invoked). The
classifier should use a `LiteLLMClient` pointed at the OpenHands proxy
(`openai/openhands/glm-5.2` with `api_base=https://llm-proxy.app.all-hands.dev`)
behind alias `kinetic-classifier-v1` — never leak the real model name.

## Next next task (Stage 2 remainder)
- `context/manager.py`: real token-aware truncation/summarisation (currently
  `NoopContextManager`). The classifier already accepts a `summary` kwarg that
  a future context manager can feed in.
- `secret/` module for credential handling.

## LLM backend notes
- `llm/client.py` now ships `LiteLLMClient(LLMClient)` instead of the old
  hand-written `AnthropicClient`. The `LLMClient` ABC interface is unchanged,
  so `agent/agent.py` and all agent-loop tests are untouched.
- `LiteLLMClient(model, api_key=None, api_base=None, max_tokens=4096)` lazy-
  imports `litellm` in `__init__` (keep core zero-dependency).
- Conversation history stays Anthropic-format (typed `tool_use`/`tool_result`
  blocks) in `ConversationState`; the client translates to OpenAI message
  shape (`assistant.tool_calls`, `tool`-role results) before calling
  `litellm.completion`, and parses the OpenAI response back to `LLMResponse`.
- Tool schemas translate Anthropic `{name,description,input_schema}` -> OpenAI
  `{type:"function", function:{name,description,parameters}}`.
- Tests for the client mock `litellm.completion` (see `tests/test_litellm_client.py`);
  the agent tests still mock via the `LLMClient` interface (`MockLLM`).
