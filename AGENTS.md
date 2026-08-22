# AGENTS.md — kinetic-agent-sdk

Repository-specific knowledge for OpenHands agents working on this SDK.

## Overview
`kinetic-agent-sdk` is a self-authored agent SDK that will replace Claude Agent
SDK inside the KINETIC coding agent. Architecture is inspired by OpenHands
`software-agent-sdk` but reimplemented from scratch, no copied code.

## Build / Test commands
- Install (dev): `pip install -e ".[dev]"`
- Install (llm backend, optional): `pip install -e ".[llm]"` (pulls in `litellm`)
- Run tests: `python -m pytest -q` (207 tests: 85 Stage 1+classifier +
  30 context manager + 20 security + 23 secret + 12 observability +
  16 hooks + 13 testing utils + 8 confirmation UX). NOTE: the
  litellm tests need
  the `[llm]` extra — install BOTH extras (`pip install -e ".[dev,llm]"`) or
  11 tests error.
- No build step beyond pip install.
- CI: `.github/workflows/test.yml` — minimal GitHub Actions workflow (push any
  branch + PR -> main, ubuntu-latest, Python 3.11, `pip install -e ".[dev,llm]"`,
  `pytest -q`). No secrets needed: all 207 tests run with mocked LLM/tool.
  Deferred on purpose: version matrix, dep cache, coverage, lint, CD.

## Stage status
- Stage 1 (Core): DONE — `tool`, `event`, `llm`, `conversation`, `agent` loop.
- Stage 2 (Context & Memory): DONE — classifier + FLASH/MAX routing DONE;
  `context/manager.py` real truncation-based compaction DONE and wired into
  the agent loop (see "Context manager" below); `secret/` module DONE (see
  "Secret management" below); `SummarizingContextManager` real LLM-summarised
  compaction DONE (see "Summarizing context manager" below).
- Stage 3 (Quality & Ops): DONE — `security/` (permission policy + audit log
  + secret redaction, wired into the agent loop), `observability/` (structured
  logging + run tracing), `hooks/` (lifecycle hooks, see "Hooks" below),
  `testing/` (public test utilities, see "Testing utilities" below), and the
  Confirmation UX via `ON_PERMISSION_CHECK` hooks (see "Confirmation UX"
  below). Deferred to later versions: richer policies, metrics/aggregation,
  external tracing (OTel/Jaeger).
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
- The session `GITHUB_TOKEN` (a `ghu_` OAuth token) is READ-ONLY for git
  contents: `git push` and Git Data API writes (blobs/trees/commits) return
  403 "Resource not accessible by integration" even though the repo
  permissions endpoint reports admin. Read ops (PR list/view/diff) work.
  Plan to hand the user a patch or ask for a write-capable credential.

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
- `kinetic_sdk/context/manager.py` — `ContextManager` ABC
  (`should_compact`/`compact`), `SimpleTruncateContextManager` (default),
  `NoopContextManager`, `SummarizingContextManager` (real LLM summarisation +
  truncation fallback), `ContextSummarizer` Protocol + `LLMContextSummarizer`
  (alias `kinetic-context-summarizer-v1`), `estimate_tokens` heuristic.

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

## Context manager (Stage 2 — DONE)
- `ContextManager` ABC: `should_compact(state, model_context_limit) -> bool`
  + `compact(state) -> ConversationState` (immutable-style: NEVER mutates the
  passed state, returns a new one). Old in-place `manage()` is gone.
- `estimate_tokens(text)` = `len(text)//4` heuristic (underestimates
  Vietnamese/code — documented; `tiktoken` stays an optional future extra).
- `SimpleTruncateContextManager(keep_last_tool_results=5, safety_threshold=0.8)`:
  compacts when estimate >= 80% of limit. Keeps first user message + tail
  anchored one message before the Nth-most-recent tool_result; the elided
  middle becomes ONE placeholder `"[N tin nhắn trước đó đã được rút gọn]"`.
  Conversations of <=2 messages (or fully protected) are returned unchanged.
- `Agent.__init__` gained `context_manager` (None -> SimpleTruncate default;
  pass `NoopContextManager` to disable) and `model_context_limit`
  (None -> `Agent.DEFAULT_MODEL_CONTEXT_LIMIT` = 128_000). In `_run_loop`,
  `_maybe_compact_context()` runs before EVERY LLM call; on compaction it
  swaps `agent.state` and emits `context.compacted` with
  messages_before/after/removed + estimated_tokens_after.
- `SummarizingContextManager` — real LLM summarisation with safe truncation
  fallback (see "Summarizing context manager" below).

## Summarizing context manager (Stage 2 — DONE)
- `SummarizingContextManager` extends `SimpleTruncateContextManager` so the
  kept structure (system prompt + first user message + N most recent tool
  results) is identical; only the elided middle span's replacement differs —
  a 1-2 sentence LLM summary instead of the static placeholder.
- Constructor accepts EITHER `summarizer` (any `ContextSummarizer`, e.g. the
  test `FakeSummarizer`) OR `summarizer_client` (a plain `LLMClient`, wrapped
  internally in `LLMContextSummarizer` with `max_tokens=150` by default) —
  passing both raises `ValueError`. Optional `event_bus` receives
  `context.summarization_failed`.
- `LLMContextSummarizer(alias="kinetic-context-summarizer-v1")` mirrors the
  classifier pattern: the SDK alias surfaces in logs/config, never a concrete
  model name. Prompt asks for a 1-2 sentence Vietnamese summary of the elided
  span (goal, key decisions, notable errors/tool results), `max_tokens`
  capped low (default 150).
- Safety rails: (1) the elided span is scrubbed with
  `security/redact.redact_value` BEFORE being sent to the summarizer model;
  (2) any summarizer failure (exception / empty / non-string result) falls
  back to the static truncation placeholder and emits
  `context.summarization_failed` (payload: manager, reason
  exception|empty_summary|non_string_summary, elided_messages, redacted
  error) — compaction never crashes the agent loop.
- `Agent.__init__` auto-wires its event bus into a `SummarizingContextManager`
  that has no bus of its own, so failure events share the agent's
  observability stream (the manager publishes directly, so these events carry
  no `run_id`). Default context manager stays `SimpleTruncateContextManager`;
  summarization is opt-in (extra model call has a cost).

## Secret management (Stage 2 — DONE)
- `secret/value.py` — `SecretValue`: wraps a secret string; `repr()`/`str()`
  ALWAYS return `"<SecretValue: [REDACTED]>"` (safe inside containers /
  `vars(obj)` dumps), `reveal()` returns the plaintext (call ONLY where the
  secret is consumed, e.g. building the HTTP request), `__eq__` compares
  against other `SecretValue` (never equal to a bare str), `__hash__` raises
  `TypeError` by design (value-derived hash would be brute-forceable).
- `secret/provider.py` — `SecretProvider` ABC (`get(key) -> str | None`),
  `EnvSecretProvider` (default, reads `os.environ`), `DictSecretProvider`
  (inject a dict; tests / SDK users with their own secret manager).
- `secret/registry.py` — `SecretRegistry(providers)` tries providers in
  order, first hit wins; `resolve(key, required=True) -> SecretValue | None`
  raises `SecretNotFoundError` (message names the missing key) when required
  and absent. Default registry = env only.
- Wiring: `LiteLLMClient(api_key=...)` and `LiteLLMClassifier(api_key=...)`
  accept `str | SecretValue | None` — plain strings are auto-wrapped
  (backward compatible, old callers unchanged). The key is stored ONLY as
  `SecretValue` on the instance and `.reveal()`ed inside `_build_request` at
  the moment the litellm request is built. `LiteLLMClassifier` also accepts
  `secrets: SecretRegistry` to resolve `OPENHANDS_API_KEY` from custom
  providers (default: env).
- Distinct from `security/redact.py`: `secret/` manages the credential
  lifecycle in memory; `redact.py` only scrubs secrets out of log text.
- NOT done (later versions): cloud secret managers (Vault/AWS SM), rotation/
  expiry.

## Security (Stage 3, part 1 — DONE)
- `security/policy.py` — `PermissionPolicy` ABC (`check(tool_name, tool_input)
  -> PermissionDecision`), `PermissionDecision(allowed, reason,
  requires_confirmation)`, `AllowListPolicy` (deny-by-default; per-tool
  `require_confirmation_patterns` are regex-with-substring-fallback matched
  against the JSON-serialised input), `PermissivePolicy` (dev/test only,
  docstring warns "KHÔNG dùng trong production").
- `security/redact.py` — `redact_secrets(text)` + `redact_value(value)`
  (recursive). Scrubs GitHub tokens (`ghp_`/`ghu_`/`gho_`/`ghs_`/`ghr_`/
  `github_pat_`), `sk-*` keys, AWS `AKIA*`, plus a keyword heuristic
  (`api_key|password|passwd|secret|token|key` followed by a 20+ char token).
  Placeholder is `[REDACTED]`.
- `security/audit.py` — `AuditLogger` base (`log_tool_call`/`log_tool_result`/
  `log_permission_denied`; entries get uuid `id`, ISO timestamp, event type,
  redacted fields), `InMemoryAuditLogger` (`.entries` list) and
  `JSONLAuditLogger(path)` (one JSON object per line, flush per write,
  context-manager support).
- `Agent.__init__` gained `permission_policy` (None -> EMPTY `AllowListPolicy`,
  i.e. deny-by-default — SDK users must opt tools in) and `audit_logger`
  (None -> `InMemoryAuditLogger`). `_execute_one` checks the policy and logs
  BEFORE executing; denial returns an error `ToolResult` to the model and
  emits `security.permission_denied`. `requires_confirmation=True` is denied
  in automated mode ("requires manual confirmation, not yet supported...") —
  extension point for a real confirmation UX later. `agent.tool_call_finished`'s
  `output_preview` is redacted before hitting the event bus.
- GOTCHA: because the default policy denies everything, old agent-loop tests
  that execute real tools now pass `permission_policy=PermissivePolicy()`
  explicitly (see `tests/test_agent.py`, `tests/test_classifier_and_routing.py`).
  New tests that exercise the loop should do the same, or test the denial
  path on purpose like `tests/test_security.py`.

## Observability (Stage 3, part 2 — DONE)
- `observability/logger.py` — `ObservabilityLogger` ABC with `attach(bus)` /
  `detach(bus)` (subscribes `handle` to the bus wildcard `"*"`, so ALL event
  types are captured without enumeration) and `build_entry(event)` producing
  `{timestamp (ISO, tz-aware), event_type, run_id, payload}`. Payloads are
  scrubbed by reusing `security/redact.redact_value` — never reimplement
  redaction here. `ConsoleObservabilityLogger` prints
  `[timestamp] [event_type] payload-json` per line (stream resolved at emit
  time so pytest `capsys` works). `InMemoryObservabilityLogger` keeps
  `.entries` and offers `get_events(event_type=None)` filtering — the test
  workhorse.
- `observability/trace.py` — `RunTrace(run_id, events)` dataclass;
  `RunTrace.collect(entries, run_id)` filters a logger's entries for one run.
  `duration()` = run_started -> run_finished (0.0 if a boundary is missing),
  `tool_calls()` from `agent.tool_call_finished`, `mode_transitions()` from
  `agent.classified` + `agent.escalated`, `to_summary()` = run_id, duration,
  tool_call_count/failed, final_mode, escalated/permission_denied/
  context_compacted flags.
- Wiring: `Agent.__init__` gained `observability_logger` (None default =
  observability fully OFF, no subscription/overhead); when given it is
  attached in `__init__` (not `run()`) so `agent.run_started` is captured.
  `Agent.run()` stamps `self._run_id = uuid4()` before classification, and
  `_emit` injects `run_id` into every event payload (new field only — old
  payload shape untouched). `agent.run_id` property exposes the current/last
  run id.
- NOT done (later): OpenTelemetry/Jaeger export, metrics/aggregation.

## Hooks (Stage 3, part 3 — DONE)
- `hooks/base.py` — `HookPoint` enum (`BEFORE_RUN`, `AFTER_RUN`,
  `BEFORE_LLM_CALL`, `AFTER_LLM_CALL`, `BEFORE_TOOL_CALL`, `AFTER_TOOL_CALL`,
  `ON_PERMISSION_CHECK`, `ON_ERROR`), `HookContext` (ONE shared dataclass,
  optional fields; which are populated is documented per HookPoint member),
  `HookResult(should_continue=True, modified_context=None)`, `Hook` Protocol
  (any callable `(HookContext) -> HookResult | None`; `None` = pure observer).
- `hooks/registry.py` — `HookRegistry(event_bus=None)`; `register` (same hook
  twice = no-op) / `unregister` / `hooks_for` / `trigger(point, context) ->
  list[HookResult]` runs hooks in registration order. A raising hook NEVER
  crashes the agent: caught, logged, emitted as `hooks.error` (payload: hook
  name, point, redacted error) on the bus, remaining hooks still run.
- Wiring: `Agent.__init__` gained `hooks` (None default = no hooks, zero
  overhead); the agent's bus is wired into a registry that has none.
  `_trigger_hooks` is the single guard. Trigger points: `run()` start/end
  (BEFORE_RUN before classification, AFTER_RUN before `agent.run_finished`),
  around every `_call_llm`, around every tool execution, ON_ERROR before
  re-raise in `run()`.
- Semantics: `should_continue=False` is honoured at BEFORE_TOOL_CALL (call
  cancelled, error ToolResult to the model, AFTER_TOOL_CALL skipped) and at
  ON_PERMISSION_CHECK (stays denied); ignored elsewhere.
  `modified_context={"tool_input": {...}}` at BEFORE_TOOL_CALL replaces the
  input BEFORE the policy check — the replacement is what gets checked,
  audited and executed. AFTER_TOOL_CALL fires only after real execution
  (not on denial/cancellation).

## Confirmation UX (Stage 3 — DONE, debt from security/ cleared)
- `_execute_one`: `requires_confirmation=True` no longer auto-denies when
  hooks exist — `_confirmed_by_hooks` triggers `ON_PERMISSION_CHECK` with
  tool_name/tool_input/permission_decision; ANY hook returning
  `should_continue=True` confirms and the call executes.
- Safe fallback unchanged: no hooks configured / none registered at that
  point / all decline (False or None) / hook raises → deny with the SAME
  historical message ("requires manual confirmation, not yet supported in
  automated mode (...)"). A raising confirmation hook thus fails CLOSED.
- SDK core ships no concrete confirmation UI on purpose; an `input()`-based
  CLI example lives in the `security/__init__.py` docstring.

## Testing utilities (Stage 3, part 4 — DONE)
- `testing/mocks.py` — `MockLLMClient(LLMClient)` (scripted list of
  `LLMResponse` or callables `(messages, tools, system) -> LLMResponse`;
  empty script -> empty `end_turn`; records `.calls`), `text_response` /
  `tool_response` builders, `MockTool(Tool)` (fixed `result` — a ToolResult
  verbatim incl. error path, any other value wrapped as output — or
  `handler(**params)`; both together -> ValueError; records `.calls`).
- `testing/assertions.py` — `assert_tool_called(trace, name, times=None)`,
  `assert_mode(trace, AgentMode | str)`, `assert_no_permission_denied(trace)`;
  all read `RunTrace`/`to_summary()`, no event parsing reimplemented.
- `testing/__init__.py` docstring = minimal end-to-end example (MockLLMClient
  + MockTool + PermissivePolicy + InMemoryObservabilityLogger + RunTrace).
- `tests/_helpers.py` now aliases `MockLLM = MockLLMClient` and re-exports
  the builders from `kinetic_sdk.testing` (no duplicated mock code);
  `EchoTool`/`FailingTool` stay test-only there.
- GOTCHA: hooks/assertion helpers registered via lambdas must return None —
  a lambda returning a tuple/value (e.g. `lambda ctx: (a.append(x), b())`)
  is collected as a HookResult and breaks `should_continue` checks.

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
