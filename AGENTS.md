# AGENTS.md — kinetic-agent-sdk

Repository-specific knowledge for OpenHands agents working on this SDK.

## Overview
`kinetic-agent-sdk` is a self-authored agent SDK that will replace Claude Agent
SDK inside the KINETIC coding agent. Architecture is inspired by OpenHands
`software-agent-sdk` but reimplemented from scratch, no copied code.

## Build / Test commands
- Install (dev): `pip install -e ".[dev]"`
- Install (llm backend, optional): `pip install -e ".[llm]"` (pulls in `litellm`)
- Install (real tokenizer, optional): `pip install -e ".[tokens]"` (tiktoken)
- Run tests: `python -m pytest -q` (786 unit tests + 4 integration deselected
  by default — see "Integration tests" below). NOTE: the litellm tests need
  the `[llm]` extra — install BOTH (`pip install -e ".[dev,llm]"`) or 11
  tests error. The tiktoken tests `importorskip` without `[tokens]`.
- Lint: `ruff check kinetic_sdk tests` — rule set DELIBERATELY narrow
  (`F`, `E9`, `I` only, configured in `pyproject.toml`); broader style rules
  are a future decision, do not silently expand.
- Type check: `mypy` (config in `pyproject.toml`, checks `kinetic_sdk` only).
  Strictness level: DEFAULT (non-strict) mode — the SDK is fully annotated
  and ships `py.typed` (PEP 561), but strict mode flags deliberate patterns
  (kwargs-dispatched `Tool.execute` overrides carry targeted
  `# type: ignore[override]`, JSON payloads are `Any`-typed). `warn_unused_ignores`
  is intentionally OFF: a few pre-existing ignores are version-dependent
  (needed on some typeshed/mypy combos, unused on others).
- No build step beyond pip install.
- CI: `.github/workflows/test.yml` — four jobs: `lint` (ruff), `typecheck`
  (mypy), `test` (matrix Python 3.10/3.11/3.12,
  `pip install -e ".[dev,llm,tokens]"`, `pytest -q`) and `integration`
  (real LLM, `continue-on-error: true`, reads `OPENHANDS_API_KEY` from
  GitHub Secrets — NON-BLOCKING by design, see "Integration tests" below).
  Unit-test jobs need no secrets: all tests run with mocked LLM/tool
  (MCP tests use fake stdio/SSE servers, no real Unity/GitHub server).
  Deferred on purpose: dep cache, coverage gate, CD.

## Integration tests (real LLM — added in the round-2 feedback pass)
- `tests/integration/test_llm_integration.py` calls a REAL provider through
  `LiteLLMClient` (no `litellm.completion` mock): simple chat round-trip,
  real tool-call decision + Anthropic↔OpenAI translation round-trip with a
  real `tool_result`, streaming round-trip, and a full `Agent.run` loop.
- Marked `pytest.mark.integration` and DESELECTED by default via
  `addopts = "-m 'not integration'"` in `pyproject.toml` — run explicitly
  with `pytest -m integration` (a CLI `-m` overrides the addopts filter).
  The module also self-skips when no API key is present.
- Config via env: `KINETIC_INTEGRATION_API_KEY` (falls back to
  `OPENHANDS_API_KEY`), `KINETIC_INTEGRATION_MODEL` (default
  `openai/openhands/glm-5.2`), `KINETIC_INTEGRATION_API_BASE` (default the
  OpenHands proxy). A key that is SET but INVALID fails the tests (correct
  behaviour — only a MISSING key skips).
- CI job `integration` is `continue-on-error: true`: rate limits / provider
  outages / expired keys must not block a PR whose unit tests pass.

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
- Stage 4 (Extensions): DONE — toàn bộ 7 module: `git/` (GitTool),
  `workspace/` (Workspace), `profiles/` (presets), `mcp/` (MCP client +
  server), `skills/` (skill discovery + vetting), `plugin/` (dynamic
  loading + static scanning) và `subagent/` (delegation + lưới an toàn chi
  phí) — xem "Stage 4 modules", "MCP", "Skills", "Plugin" và "Subagent"
  bên dưới.

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

## Stage 4 modules (git/, workspace/, profiles/ — DONE)
- `git/tool.py` — `GitTool(Tool)`: curated git sub-actions via one `action`
  param (`status`, `diff`, `add`, `commit`, `branch`, `checkout`, `push`,
  `pull`, `log`); NOT a full git wrapper (no rebase/cherry-pick/submodule).
  All commands run as argv lists (never `shell=True`), refs starting with
  `-` are rejected (option-injection guard), `add` always inserts `--`, and
  `log` caps `max_count` at `MAX_LOG_COUNT` (100, default 20) so history
  can't flood the context. Constructor: `workdir` (default cwd),
  `secrets: SecretRegistry` (default env registry — the module never reads
  `os.environ` directly), `remote_token_key` (default `GIT_TOKEN`),
  `timeout`, and an injectable `runner` (default `subprocess.run`) so tests
  never spawn real git. push/pull resolve the token via the registry, reveal
  it only when building `-c http.extraHeader=AUTHORIZATION: bearer ...`, and
  scrub the literal value + `redact_secrets` from all output/metadata before
  returning. Missing token is NOT an error (git falls back to its own
  credential config).
- Force-push safety: the JSON-serialised input of a force push contains
  `"force": true`; `GitTool.REQUIRE_CONFIRMATION_PATTERNS` is a ready-made
  pattern list for `AllowListPolicy(require_confirmation_patterns={"git":
  GitTool.REQUIRE_CONFIRMATION_PATTERNS})` so force pushes require
  confirmation (via ON_PERMISSION_CHECK hooks; denied without one). The
  result metadata also records `force=True`. GitTool is an ordinary Tool —
  no bypass of `permission_policy` (test asserts the default empty
  AllowListPolicy denies it).
- `workspace/manager.py` — `Workspace(root_path)` (root must be an existing
  dir, canonicalised once) with `resolve(relative_path)` (realpath +
  commonpath containment check; `../`, absolute paths outside root and
  symlink escapes all raise `PathTraversalError`) and
  `list_files(pattern=None)` (sorted root-relative POSIX paths, files only,
  `fnmatch` filter where `*` crosses directories). One agent, one workspace
  this version; migrating existing tools to route paths through it is
  deferred (GitTool pairs via `GitTool(workdir=workspace.root_path)`).
- `profiles/presets.py` — `dev_profile()` (PermissivePolicy +
  InMemoryAuditLogger + ConsoleObservabilityLogger) and
  `production_profile(*, audit_log_path, allowed_tools=None,
  summarizer_client=None)` (AllowListPolicy deny-by-default, git
  confirmation patterns pre-loaded when "git" is allowed, JSONLAuditLogger,
  SummarizingContextManager only when a summarizer client is given). Both
  return plain kwargs dicts for `Agent(**profile)` — they NEVER construct an
  Agent themselves (no import-time side effects); docstrings stress they are
  starting points, not the one true config.

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

## MCP (Stage 4 — DONE)
- `mcp/protocol.py` — JSON-RPC 2.0 message layer: `JsonRpcRequest`/
  `JsonRpcResponse`/`JsonRpcNotification` dataclasses, `encode` (one message
  = one JSON line ending `\n` — newline-delimited, NOT LSP Content-Length
  framing), strict `decode` (malformed JSON / missing `jsonrpc: "2.0"` /
  missing fields / response with both or neither of result+error all raise
  `MCPProtocolError` — never silently degrade to None), thread-safe
  `RequestIdGenerator` (monotonic int ids).
- `mcp/transport.py` — `Transport` ABC (`send`/`receive(timeout)`/`close`,
  context manager). `StdioTransport(command, args, env, process_factory)`
  spawns a subprocess (factory injectable; `env=None` inherits os.environ, a
  dict REPLACES it); a daemon pump thread feeds stdout lines into a queue so
  `receive(timeout)` works on blocking pipes and raises `MCPTimeoutError`;
  child death (EOF) surfaces as `MCPTransportError` with exit code + stderr
  tail, and STAYS raised on later receives. `StdioTransport.attach(reader,
  writer)` is the SERVER-side entry (protocol on own stdin/stdout;
  `close()` only flushes — streams belong to the host). `SSETransport(url,
  headers, connect_timeout, read_timeout)`: lazy `_open()` on first
  send/receive reads the mandatory `endpoint` event first (connect_timeout
  bound), `send` POSTs JSON to that endpoint (new connection per POST,
  expects 200/202), `receive` reads `message` events off the SSE stream.
  Both transports have `__del__` best-effort cleanup (no orphan subprocess
  when users skip `with`).
- `mcp/client.py` — `MCPClient(transport, init_timeout=10, request_timeout=30)`
  is transport-agnostic. `initialize()` runs the mandated 3-step handshake
  (initialize request -> response validated for a string `protocolVersion`,
  else `MCPHandshakeError` -> `notifications/initialized` notification) and
  is idempotent; `list_tools()`/`call_tool()` raise `MCPClientError` if
  called before it (`_initialized` flag). `_request` correlates responses by
  id (mismatch raises, never silently pairs), skips interleaved
  notifications (optional `on_notification` callback), ignores
  server-initiated requests (sampling/roots unsupported this version), and
  raises `MCPServerError(code, message, data)` for JSON-RPC errors.
- `mcp/adapter.py` — `MCPToolAdapter(Tool)`: wraps ONE server tool; the
  registered name is ALWAYS `"{server_name}.{tool_name}"` (e.g.
  `"unity.search"`) so two servers' same-named tools never collide;
  `from_mcp_schema` maps MCP's camelCase `inputSchema` -> Kinetic
  `parameters`. `execute()` maps every failure mode (transport dead,
  `MCPServerError`, `isError: true`, malformed result) to
  `ToolResult(error=...)` — never raises into the agent loop — and scrubs
  ALL output through `redact_value` (external server output is untrusted).
- `mcp/registry.py` — `MCPServerConfig.stdio(command, args, env)` /
  `.sse(url, headers)` (exactly one flavour, XOR enforced);
  env/headers values may be `str | SecretValue` — plaintext revealed ONLY at
  transport-build time, module never reads `os.environ` (same rule as
  `git/`). `MCPServerRegistry(secrets=...)`: `register` stores config
  without connecting; `connect(name)` spawns transport + handshake lazily,
  caches the client, and closes the half-open transport if the handshake
  fails (no orphans); `get_tools_as_kinetic_tools(name)` wraps every server
  tool in an adapter with the server-name prefix.
- `mcp/server.py` — `MCPServer(tools, permission_policy, audit_logger)` is
  direction B: Kinetic AS an MCP server. `handle_message` is transport-free
  (unit-testable); it answers `initialize`, refuses `tools/list`/
  `tools/call` with `-32002` until `notifications/initialized` arrives, and
  routes EVERY `tools/call` through `permission_policy.check` + audit log
  exactly like the internal agent loop (deny -> MCP `isError` result;
  `requires_confirmation` -> denied, same safe fallback message; unknown
  tool -> `-32602`; unknown method -> `-32601`). Tool output is redacted
  before serving. `serve_forever(transport)` loops until the peer hangs up
  (clean shutdown signal), answering malformed lines with `-32700` and
  continuing. `MCPServer.serve_stdio(tools, policy)` is the entry point when
  Kinetic is spawned as an MCP subprocess (stdout = protocol channel, never
  print to it).
- Example config (no real Unity server needed for tests — the test fakes in
  `tests/_mcp_fakes.py` speak just enough MCP over real pipes/sockets):

  ```python
  from kinetic_sdk.mcp import MCPServerConfig, MCPServerRegistry
  from kinetic_sdk.secret import SecretRegistry

  secrets = SecretRegistry()  # env-based
  registry = MCPServerRegistry(secrets=secrets)
  registry.register(
      "unity",
      MCPServerConfig.stdio(
          command="unity-mcp",
          args=["--project", "./Game"],
          env={"UNITY_API_KEY": secrets.resolve("UNITY_API_KEY")},
      ),
  )
  tools = registry.get_tools_as_kinetic_tools("unity")  # ["unity.search", ...]
  agent = Agent(llm=..., tools=[*tools, GitTool()], ...)
  ```
- GOTCHA: `StdioTransport.attach()` does NOT close the streams on `close()`
  (they belong to the host process) — tests driving `serve_forever` over
  `os.pipe()` must close the client writer themselves to send the EOF that
  ends the server loop (see `tests/test_mcp_server.py`).
- NOT done (later versions): tool-list caching across runs, concurrent/async
  calls (id correlation is already spec-correct), server-initiated requests
  (sampling/roots), resources/prompts MCP capabilities, config UI/CLI.

## Skills (Stage 4 — DONE)
- `skills/skill.py` — `Skill` frozen dataclass (name, description, source,
  root_path, version). `from_directory(dir, source)` parses SKILL.md
  frontmatter with a hand-rolled flat `key: value` parser (no pyyaml dep);
  name must match `^[a-z0-9]+(-[a-z0-9]+)*$` (max 64 chars) AND equal the
  directory name; description non-empty, truncated at
  `MAX_DESCRIPTION_LENGTH = 1024` with a notice instead of rejected.
  `read_main()` returns the body (lazy, uncached — a deleted directory raises
  `SkillParseError` immediately). `list_resources(pattern, resource_type)`
  and `read_resource()` are HARD-SCOPED to `scripts/`/`references/`/`assets/`
  (exposure layer — deliberately narrower than the zip extension allowlist,
  which is the extraction layer; the two filters are independent). All reads
  go through `Workspace(root_path)` — no hand-rolled traversal checks.
- `skills/loader.py` — `SkillLoader` ABC + `FileSystemSkillLoader(root_dir,
  source_label="local")`: one level deep, sorted, skips non-dirs / missing
  SKILL.md silently, parse errors and symlink-duplicate realpaths warn+skip
  (one broken skill never fails the scan). Discovery is metadata-only
  (progressive disclosure) — bodies/resources are read lazily later.
- `skills/zip_loader.py` — `ZipSkillLoader(zip_path, max_total_size_bytes=
  10MB, max_entries=500, allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS)`:
  context manager; temp dir lives until `close()` (discover() caches, second
  call returns cached, discover-after-close raises RuntimeError). Extraction
  checks entries one by one (NEVER `extractall()`): zip-slip raises
  `ZipSkillError` (whole archive poisoned); symlink entries and disallowed
  file extensions warn+skip (one bad entry never fails the archive);
  directory entries bypass the extension allowlist (they have no extension);
  declared-size sum pre-check + running-bytes guard against forged headers.
  GOTCHA (lifecycle, hit twice during spec review): `Skill` objects from
  `discover()` point into the temp dir — after `close()` they raise on read.
  Callers wanting long-lived skills MUST use `add_skill_from_zip` (below),
  which materialises vetted skills into `persist_dir` while the temp dir is
  still alive; the registry never holds a temp-dir-backed skill.
- `skills/registry.py` — `SkillRegistry` with `add`/`add_many`/`load_from`
  (NOT constructor-injected loaders): the ONLY acceptance gate. Invariant:
  `source != "local"` requires a `VetResult` with `clean=True`, else
  `SkillVetError` — no bypass path exists. First registration wins on name
  collision (warn+skip, no overwrite). `get`/`load(name)` (`load`, not
  `invoke` — skills are read text, not executed actions),
  `to_prompt_catalog()` = sorted `- name: description` lines.
- `skills/vet.py` — `VetFlag`/`VetResult`, `StaticSkillScanner` (regex
  categories prompt_injection / auto_exec / credential_exfil /
  permission_bypass as critical, suspicious_url as warning, hidden Unicode —
  Tag block U+E0000–E007F, zero-width, C0 controls — as critical),
  `LLMSkillReviewer(llm, max_tokens=200)` (separate judge client, fixed JSON
  schema, content redacted via `redact_value` before sending; ANY failure →
  fail-safe `review_failed` WARNING flag, clean stays True), and
  `vet_skill(skill, llm_reviewer=None, raise_on_critical=True)` as the single
  policy entry point. Two iron rules: (1) `clean` is ALWAYS re-derived from
  parsed `severity` — the model's own `clean` field is never trusted
  (critical => not clean, whatever the model claims); reviewer output is
  untrusted data, never interpolated/executed. (2) `raise_on_critical=False`
  only suppresses the exception — the result is still `clean=False` and the
  registry still rejects it; the two mechanisms are fully independent.
  LLM review is skipped when static scan already found a critical (no wasted
  call). Static scanner intentionally has NO negation analysis — negated
  warnings ("Never read ~/.ssh/...") false-positive by design; disambiguation
  is the LLM reviewer's job (documented in the class docstring).
- `skills/__init__.py` — `add_skill_from_zip(registry, zip_path, persist_dir,
  llm_reviewer=None, raise_on_critical=True)`: discover → vet each skill →
  `shutil.copytree` pass-vet skills into `persist_dir/<name>/` (destination
  resolved through a `Workspace` over persist_dir; existing destination →
  `FileExistsError`, never silent overwrite) → `registry.add` the PERSISTENT
  `Skill` (source stays `"zip"`). All temp-dir access happens inside one
  `with ZipSkillLoader(...)` block.
- Loading a skill NEVER grants tools/permissions — a skill is instructional
  text; anything it tells the agent to do still passes `permission_policy`.
- NOT done (later versions): GitHub/remote loaders (pin by commit SHA,
  `fetch(...) -> (path, resolved_sha)`), trigger-based auto-injection,
  persistent installed-state manager, `mcp_tools` frontmatter (rejected on
  principle — skills must not self-grant tools).

## Plugin (Stage 4 — DONE)
- `plugin/` = discover + load động các package Python bên ngoài để chúng
  đăng ký thêm `Tool` vào Kinetic. Đây KHÔNG phải kiến trúc "everything is
  a plugin": agent loop, `permission_policy`, `EventBus` giữ nguyên —
  plugin chỉ đóng góp tool, và tool đó chạy qua `permission_policy.check`
  y hệt tool nội bộ/MCP (có test tích hợp verify cả chiều allow lẫn deny
  trong `tests/test_plugin_registry.py`).
- **GIỚI HẠN BẢO MẬT — scanner là TRIPWIRE, KHÔNG phải sandbox.** Plugin là
  code Python chạy TRỰC TIẾP trong cùng process với Kinetic: khác MCP
  server (subprocess cách ly — worst case là ngắt kết nối) và khác skill
  (text trơ — agent vẫn phải "đọc rồi quyết định"), plugin độc hại có thể
  đọc `os.environ`, mở socket, hay monkey-patch `kinetic_sdk.security` để
  vô hiệu hoá permission check của mọi request SAU nó mà không cần lừa ai.
  Python không có sandbox thật in-process, nên static scanner chỉ bắt lỗi
  vô tình + tấn công không tinh vi và tăng chi phí tấn công — kẻ cố tình
  né (obfuscation, decode string lúc runtime) vẫn lách được. "Plugin qua
  scan" TUYỆT ĐỐI không có nghĩa "plugin an toàn"; chỉ load plugin từ nguồn
  đáng tin cậy ngang credential của chính bạn. Mọi docstring/log message
  trong module đều phải giữ đúng sự thật này — không được quảng cáo
  scanner như sandbox.
- `plugin/manifest.py` — `PluginManifest` (frozen): name (tái dùng
  `SKILL_NAME_PATTERN` + `MAX_NAME_LENGTH` từ skills), version (optional),
  entry_point dạng `"module.sub:factory"` (validate bằng regex), source
  (`"entry_point"` | `"directory"` — do loader gán, plugin không tự xưng),
  `declared_capabilities: frozenset[str]` (subset của
  `KNOWN_CAPABILITIES = {"tool","hook"}`; chỉ "tool" được implement, khai
  "hook" bị `PluginLoadError` rõ ràng), `directory` (bắt buộc khi source =
  directory, cấm khi source = entry_point). Directory plugin đọc frontmatter
  `PLUGIN.md` (tái dùng `_split_frontmatter` của skills — cùng 1 parser
  khỏi lo drift); name phải khớp tên thư mục như skill.
- `plugin/discovery.py` — gộp 2 nguồn: entry points
  (`importlib.metadata.entry_points(group="kinetic_sdk.plugins")`,
  `entry_points_provider` inject được cho test) + directory convention
  (quét 1 cấp, sorted, dir thiếu PLUGIN.md skip lặng, PLUGIN.md hỏng
  warn+skip — y hệt FileSystemSkillLoader). Trùng tên giữa 2 nguồn →
  `PluginManifestError` NGAY ở discover, trước khi import bất cứ thứ gì.
  Discovery TUYỆT ĐỐI không import code plugin (progressive disclosure, có
  test bằng side-effect marker).
- `plugin/scanner.py` — AST-based (KHÔNG regex trên text), visitor resolve
  alias import (`from subprocess import run as r; r(...)` bị bắt y như
  `subprocess.run(...)`). Critical: `eval/exec/compile` với input
  non-literal; `os.system/os.popen` + `subprocess.Popen/run/call/
  check_call/check_output` gọi trực tiếp (GitTool wrapper KHÔNG bị chặn);
  truy cập `os.environ` mọi dạng (attribute, `from os import environ`,
  `getattr(os,"environ")`) — phải qua SecretRegistry; GHI vào namespace
  `kinetic_sdk.security.*` (assign/delete/setattr — đọc API công khai thì
  OK); `__import__`/`importlib.import_module` với tên non-constant.
  Warning: import socket/urllib/http/requests/httpx/aiohttp (network trực
  tiếp ngoài kênh mcp/); `open()` với path non-literal. Source không parse
  được = critical "unparseable". Policy nằm ĐÚNG 1 chỗ:
  `scan_plugin(manifest) -> ScanResult`; loader chỉ đọc `ScanResult.clean`
  (tái dùng `VetFlag` của skills/vet, `location` = `path:lineno`).
  GOTCHA: locate module entry_point-source dùng `PathFinder.find_spec`
  walk từng cấp (`_find_spec_no_import`) — KHÔNG dùng
  `importlib.util.find_spec` vì hàm đó import parent package = chạy code
  plugin TRƯỚC scan. Directory source: scan MỌI file `*.py` dưới plugin
  root, không chỉ entry module.
- `plugin/loader.py` — `PluginLoader(audit_logger=None)`. Thứ tự cố định:
  scan (không clean → `PluginVetError`, KHÔNG import) → import → gọi
  factory → kết quả PHẢI là `list[Tool]` (sai kiểu → `PluginLoadError`,
  không coerce; factory trả `PluginManifest` thì follow đúng 1 hop —
  entry-point plugin làm manifest provider) → validate từng phần tử là
  `Tool`. MỌI exception lúc import/factory bị bọc `PluginLoadError` kèm
  tên plugin + `__cause__` — 1 plugin hỏng không kéo sập plugin khác.
  Directory module import dưới tên `kinetic_plugin_<name>.<module>` (không
  bao giờ shadow package thật trong sys.modules), parent package dựng sẵn
  để relative import trong plugin chạy được mà không đụng `sys.path`;
  import lỗi thì dọn sys.modules. Audit MỌI lần load pass lẫn fail qua
  `AuditLogger.log_event("plugin_load", ...)` (method MỚI thêm vào
  `security/audit.py` cho non-tool events, field values redacted): plugin,
  source, outcome (loaded/rejected/failed), toàn bộ scan flags (kể cả
  warning), danh sách tool đăng ký.
- `plugin/registry.py` — `PluginRegistry(directory, entry_points_provider,
  audit_logger, loader, reserved_names)`: `discover()` +
  `load_all(on_error="skip"|"raise")` (mặc định skip+log; "raise" =
  fail-fast). Trùng tên TOOL giữa 2 plugin, hoặc plugin tool đụng
  `reserved_names` (tool nội bộ/MCP) → `PluginLoadError` luôn raise, kể cả
  skip mode. Lý do REJECT thay vì prefix như `mcp/adapter.py`: tool MCP
  prefix vì 2 server độc lập hợp lệ khi trùng tên generic; plugin tool
  chạy in-process dưới cái tên tác giả tự chọn — trùng = lỗi packaging
  hoặc cố shadow tool khác trong mắt model, đổi tên ngầm sẽ làm model gọi
  sai.
- NOT done (later versions): unload/hot-reload, `hook` capability cho
  plugin (chỗ mở rộng đã để sẵn trong KNOWN_CAPABILITIES), sandbox thật
  (subprocess/container — hạn chế đã biết trước, ghi rõ trong docs, không
  phải việc "khắc phục ngay").

## Subagent (Stage 4 — DONE)
- `subagent/` = agent TỰ SINH agent khác dùng chính tool/credential của
  mình — module duy nhất của Stage 4 có rủi ro runaway recursion đốt tiền
  API thật và rò rỉ output trung gian nhạy cảm ngược vào context cha.
- **3 QUYẾT ĐỊNH THIẾT KẾ ĐÃ CHỐT (KHÔNG phải thiếu sót, đừng "sửa"):**
  1. **Kế thừa toàn bộ, không thu hẹp.** Sub-agent nhận NGUYÊN tool list +
     CÙNG object `permission_policy` của cha (giống Claude Agent SDK). Thu
     hẹp cứng sẽ phá pattern hợp lệ "controller hẹp → executor cần tool
     rộng hơn để làm việc thật" (đã có tiền lệ lỗi thật trên OpenCode khi
     làm ngược lại). Điểm khác DUY NHẤT so với "sao chép y hệt cha": mỗi
     sub-agent có `system_prompt` RIÊNG — bắt buộc non-empty trong
     `SubagentSpec`, không bao giờ dùng lại system prompt của cha.
  2. **KHÔNG giới hạn cứng số tầng lồng nhau** (giống OpenHands):
     delegation là 1 tool bình thường (`DelegateTool`), không đặc biệt hoá
     số tầng trong `Agent.run()`.
  3. **Thay giới hạn tầng bằng 2 lưới an toàn theo CHI PHÍ/HÀNH VI**
     (mô hình oh-my-opencode): `SpawnBudget` — 1 instance CHIA SẺ cho toàn
     cây (thread-safe bằng lock, sẵn sàng cho parallel sau này), mọi tool
     call của mọi agent trong cây đều trừ chung 1 ngân sách (default 4000),
     hết → `BudgetExceededError`; và `RepetitionCircuitBreaker` — RIÊNG
     từng agent (không cộng dồn cây), cùng (tool_name + arguments serialize
     ổn định `json.dumps(sort_keys=True, default=str)`) quá N lần liên
     tiếp (default 20) → `RepetitionLimitError`. Đổi tool hoặc đổi args là
     reset bộ đếm.
- `subagent/manifest.py` — `SubagentSpec` frozen dataclass (name theo đúng
  `SKILL_NAME_PATTERN`/`MAX_NAME_LENGTH` của skills, `system_prompt` bắt
  buộc, `description` cho cha quyết định khi nào delegate, `model` optional
  để dùng model rẻ hơn). `model` khác model cha CHỈ có tác dụng khi caller
  truyền `llm_factory(model) -> LLMClient` — SDK không tự biết cách dựng
  provider client generic; thiếu factory → `SubagentError`, không lặng lẽ
  chạy model cha.
- `subagent/delegation.py` — `spawn_subagent(parent, spec, budget)` trả về
  1 `Agent` MỚI: kế thừa tools/policy/audit_logger/event_bus/classifier/
  context_manager/hooks từ cha, `ConversationState` MỚI HOÀN TOÀN chỉ với
  `spec.system_prompt` (kênh cha→con DUY NHẤT là `task_prompt` lúc run;
  kênh con→cha DUY NHẤT là `DelegationResult.final_message` — transcript
  trung gian không bao giờ rò ngược). `run_subagent(...)` = spawn + run +
  trả `DelegationResult(final_message, agent_id)`, để lỗi guardrail
  propagate cho caller.
- **Cơ chế enforcement — `_GuardedLLMClient`, KHÔNG sửa `Agent.run()`:**
  LLM client của sub-agent được bọc 1 lớp guard charge MỌI tool call model
  REQUEST (kể cả call sau đó bị policy deny — turn LLM đã tốn tiền rồi)
  vào budget + breaker TRƯỚC khi thực thi. Raise xảy ra trong `chat()` —
  đường `_call_llm` mà `Agent.run` RE-RAISE chứ không nuốt (khác hook bị
  `HookRegistry` catch, khác exception trong `tool.execute` bị
  `_execute_one` thành `ToolResult`). Đây là lý do guardrail fail-fast
  thay vì chờ `max_iterations`. `DelegateTool.execute` bắt
  `BudgetExceededError`/`RepetitionLimitError`/mọi Exception và map thành
  `ToolResult(error=...)` — loop của cha không bao giờ sập.
- GOTCHA (đã bắt gặp khi implement): sub-agent spawn tiếp sub-sub-agent thì
  `parent_agent.llm` đã LÀ `_GuardedLLMClient` — phải UNWRAP (`base.inner`)
  trước khi wrap guard mới, nếu không mỗi tool call bị charge budget 2 lần
  và breaker của cha đếm lẫn call của con (phá nguyên tắc per-agent).
  `spawn_subagent` đã xử lý; đừng wrap guard chồng lớp ở chỗ khác.
- `DelegateTool` (name `"delegate"`): registry `dict[str, SubagentSpec]`
  trong constructor, PHẢI `bind(agent)` sau khi dựng agent (tool cần agent,
  agent cần tools — gài 2 bước). Khi spawn, mọi DelegateTool trong tool
  list cha được CLONE (`_clone_for(budget)`) + bind vào sub-agent mới,
  dùng CHUNG spec registry và CHUNG budget instance — đây là cơ chế khiến
  cây nhiều tầng chia sẻ 1 ngân sách. Chính DelegateTool vẫn đi qua
  `permission_policy.check` như mọi tool (deny `"delegate"` trong
  AllowListPolicy = không sub-agent nào spawn được — có test).
- Root agent (task gốc) KHÔNG được guard — tool call của chính root không
  trừ budget (budget canh "cây sub-agent phát sinh", root nằm TRÊN cây).
  Chỉ các agent do `spawn_subagent` tạo ra mới có guarded client.
- Audit: `subagent_spawn` (parent_agent_id + agent_id — mint qua
  `agent_id_for`, WeakKeyDictionary có lock) và `subagent_finished`
  (outcome: completed | budget_exceeded | repetition_limit | error) đều qua
  `AuditLogger.log_event` trên audit logger CHIA SẺ của cả cây → dựng lại
  được cây cha-con sau này. Final message của sub-agent bị
  `redact_secrets` trước khi vào context cha (chống rò rỉ bí mật/PII).
- Hạn chế đã biết (sequential execution bản này): runaway đệ quy với
  budget mặc định 4000 sẽ chạm giới hạn đệ quy Python (~1000 frames) trước
  khi hết budget — `RecursionError` vẫn được map thành `ToolResult` error,
  budget vẫn chặn tổng số spawn, không crash. Test runaway dùng budget nhỏ
  (30) để `BudgetExceededError` bắn trước. `chat_stream` của guard delegate
  thẳng không charge (agent loop chỉ dùng `chat()`).
- NOT done (later versions): chạy sub-agent song song thật (budget đã
  thread-safe sẵn), field tuỳ chọn trong `SubagentSpec` để thu hẹp tool khi
  spawn (chỗ mở rộng đã để, bản này luôn kế thừa toàn bộ), giới hạn tầng
  (đã quyết định KHÔNG làm — thay bằng budget + circuit breaker).

## Hardening pass (2026-08, post-Stage-4 — DONE)

Đợt tăng cường độ chịu lỗi runtime sau khi review toàn SDK. Mọi quyết định
dưới đây là ĐÃ CHỐT, đừng "sửa" ngược trừ khi có lý do mới.

### Correctness fixes (bug thật, đã có regression test)
- **Compaction orphan `tool_result` (parallel tool calls).** Bug gốc:
  `_tail_start` neo tail vào "1 message trước tool_result thứ N-từ-cuối" —
  với parallel calls (1 assistant chứa N `tool_use` + N message
  `tool_result`), anchor rơi GIỮA nhóm → tail giữ result mồ côi → provider
  400 ở call kế tiếp. Fix: `_expand_to_group_boundary` lùi `tail_start` cho
  tới khi mọi `tool_result` trong tail có `tool_use` tương ứng; không resolve
  được (history đã corrupt từ trước) thì trả 0 → compact no-op (an toàn,
  KHÔNG bao giờ xuất history invalid). Head (message[0]) chỉ được giữ khi
  không chứa tool block. Fixture test cũ (`_long_conversation`) từng build
  assistant text-trơn + result rời = history vốn invalid — đã sửa fixture
  thành dạng thật (assistant chứa `tool_use` blocks).
- **`chars_per_token` constructor param từng được lưu nhưng KHÔNG bao giờ
  dùng** (ABC `estimate_state_tokens` gọi heuristic default 4). Giờ estimation
  đi qua `self._count_tokens` per-manager.

### Resilience mới
- **LLM retry/timeout** (`LiteLLMClient`): `timeout` forward sang
  `litellm.completion`; `max_retries=2` mặc định (0 = hành vi cũ 1 phát 1),
  backoff `retry_base_delay * 2^k` cap `retry_max_delay` + jitter 25%.
  `_is_retryable`: status 408/409/429/5xx hoặc tên class chứa
  timeout/connection/ratelimit/unavailable; 4xx khác raise ngay (retry chỉ
  đốt quota). `chat_stream` chỉ retry lúc TẠO stream, không retry giữa
  chừng (tránh duplicate text). `LiteLLMClassifier` dùng chung client nên
  hưởng retry tự động.
- **Tool timeout** (`Agent(tool_timeout=...)`): `tool.execute` chạy trên
  thread pool dùng chung (`_get_executor`, max_workers=8); hết giờ → error
  ToolResult, loop sống tiếp. Python không kill được thread → timed-out call
  CHẠY NGẦM tiếp (ghi rõ trong docstring); tool bọc subprocess vẫn phải tự
  có timeout thật (GitTool/TerminalTool đã có).
- **EventBus**: `subscribe()` giờ WARN khi đăng ký coroutine function
  (sync `publish()` skip coroutine lặng lẽ — trước đây user không biết tại
  sao listener không chạy).

### Context & routing
- **Token counter cắm được**: `SimpleTruncateContextManager(token_counter=...)`
  + `context/tokens.py` `TiktokenCounter` (extra `[tokens]`, lazy import,
  fallback heuristic khi encode lỗi). `SummarizingContextManager` nhận
  `token_counter` qua super.
- **Cắt nội dung tool result lớn**: `max_tool_result_chars` (None = tắt,
  min 100). Kept `tool_result` vượt ngưỡng → head 2/3 + marker
  `"[... N ký tự ở giữa đã được cắt bớt ...]"` + tail 1/3. Chạy CẢ KHI
  `removed <= 0` — đây là fix cho "tail được bảo vệ nhưng tự nó đã tràn
  window" (5 build log khổng lồ) mà elision không cứu được.
- **Sticky MAX**: `_sticky_max` — run nào classify MAX (hoặc escalate) thì
  các run SAU trong cùng conversation không bao giờ rớt về FLASH (trước đây
  `_classify_and_route` chạy lại mỗi run và có thể hạ MAX→FLASH, tự thu nhỏ
  iteration budget giữa chừng). Fallback do classifier exception KHÔNG set
  sticky (lỗi tạm thời không được ghim MAX mãi). Event `agent.classified`
  thêm field `applied_mode` khi override.

### Agent loop mới
- **Persistence**: `conversation/store.py` — `ConversationStore` ABC +
  `JsonFileConversationStore` (atomic write tmp+rename, `SCHEMA_VERSION=1`,
  file corrupt/sai version → `ConversationStoreError`, KHÔNG lặng lẽ start
  fresh). `Agent(state_store=...)`: explicit `state` vẫn thắng; không thì
  load lúc init; `_persist_state()` sau MỖI TURN nhưng CHỈ ở boundary
  replay-valid (assistant text turn hoặc ngay sau tool results) — persist
  giữa chừng sẽ lưu history dangling `tool_use`, resume xong provider 400.
  Store hỏng → log warning + event `agent.state_persist_failed`, run không
  chết. File lưu RAW history (có thể chứa credential trong tool output) —
  không redact, docstring cảnh báo rõ.
- **Cancellation**: `agent.cancel()` (threading.Event, gọi từ thread khác
  được), check ở đầu mỗi iteration + ngay sau BEFORE_LLM_CALL hooks (hook
  cancel phải chặn được LLM call sắp xảy ra) + giữa tool batch. Hủy giữa
  batch: call còn lại KHÔNG execute nhưng vẫn nhận error `tool_result`
  ("skipped: run cancelled") để history không dangling. Emit
  `agent.cancelled`, `run()` trả final text tốt nhất hiện có (có thể rỗng),
  KHÔNG raise. Flag clear ở đầu `run()` mới. Tool đang chạy KHÔNG bị preempt.
- **Schema validation**: `tool/validation.py` `validate_tool_input(schema,
  params) -> list[str]` — subset JSON Schema (type/properties/required/
  items/enum/additionalProperties; KHÔNG $ref/combinators, keyword lạ bỏ
  qua). Strict ở top-level: param lạ bị reject trừ khi schema khai
  `additionalProperties: true` hoặc không khai `properties` (vì extra kwarg
  sẽ TypeError trong `execute(**params)` — giờ thành error message rõ ràng
  cho model tự sửa). bool KHÔNG phải integer. Wire trong `_prepare_call`
  (sau policy, trước execute), tắt bằng `Agent(validate_tool_inputs=False)`.
- **Parallel tool execution (OPT-IN)**: `Agent(parallel_tool_execution=True)`
  — `_execute_one` đã tách 3 phase: `_prepare_call` (hooks/policy/confirm/
  validation — main thread, tuần tự), execute (pool), `_finalize_call`
  (audit + AFTER hooks — main thread, THEO THỨ TỰ GỐC của model). Mặc định
  False = tuần tự như cũ. Tool custom phải thread-safe mới được bật.
  KHÔNG gọi `_run_tool` trong parallel path (nested pool → deadlock) —
  submit `tool.execute` trực tiếp, timeout qua `future.result(timeout)`.

### Bộ tool chuẩn (mới)
- `terminal/` — `TerminalTool(workspace=None, timeout=120, max_timeout=600,
  max_output_chars=30_000, env=None)`: `bash -c`, process-group kill khi
  timeout (start_new_session + killpg SIGKILL), stdout+stderr gộp, cắt
  head/tail. `env=None` inherit env của process (module KHÔNG đọc os.environ
  — cùng rule với git/mcp); dict thì REPLACE. KHÔNG tự chặn lệnh nguy hiểm
  (đó là việc của policy); ship sẵn `REQUIRE_CONFIRMATION_PATTERNS`
  (rm -rf/sudo/mkfs/dd/chmod -R/...) cắm vào AllowListPolicy như GitTool.
- `files/` — `FileTool(workspace, max_view_lines=2000)`: actions
  view/create/str_replace(unique match bắt buộc)/insert/undo_edit (in-memory
  1-level, redo-able). MỌI path qua `Workspace.resolve` — `../`, absolute
  ngoài root, symlink escape đều bị chặn trước khi đụng disk. Workspace
  BẮT BUỘC (file tool không biên giới = whole-filesystem tool, từ chối).

### CI/DX
- Ruff lint (`F,E9,I`) trong `pyproject.toml` + CI job riêng; test matrix
  3.10/3.11/3.12; CI install `.[dev,llm,tokens]`.
- GOTCHA ruff: `--select F --fix` xóa re-export trong `tests/_helpers.py`
  (F401 false positive) → tests sập hàng loạt. Re-export viết dạng
  `MockLLM = _mocks.MockLLMClient` (alias assignment) — ruff không đụng.
- NOT done (later): async agent loop (AsyncLLMClient vẫn interface-only),
  cost/token budget cho root run, metrics/OTel, policy theo resource, MCP
  resources/prompts + reconnect, hot-reload plugin.

## Round-2 feedback pass (2026-08, post-hardening — DONE)

Xử lý feedback từ kỹ sư test thật, 4 ưu tiên. Quyết định đã chốt:

### Streaming vào agent loop
- `Agent.run(user_message, *, stream: bool = False)` — chọn tham số keyword
  trên `run()` thay vì method `run_stream()` riêng vì ÍT PHÁ VỠ interface
  nhất: signature cũ nguyên vẹn, return type vẫn `str` (final text), caller
  nhận real-time qua event bus.
- Khi `stream=True`: `_call_llm` consume `llm.chat_stream()`, mỗi text chunk
  emit `agent.text_delta` (payload `{"delta": str}`) NGAY KHI NHẬN, response
  tổng hợp từ event `done` đi tiếp vào loop như cũ → tool calling/escalation
  không đổi (model stream text trước rồi gọi tool vẫn đúng thứ tự — có test).
  Client không hỗ trợ streaming (`NotImplementedError`) → fallback `chat()`
  kèm warning log, KHÔNG crash.
- `LiteLLMClient.chat_stream` ĐÃ SỬA aggregation tool call: trước đây giữ
  fragment đầu tiên rồi dedup → args bị cắt với provider stream thật. Giờ
  accumulate theo `index` (id/name ở chunk đầu, arguments nối dần), parse
  JSON MỘT LẦN ở cuối (`_tool_call_from_fragments`). Provider gửi call hoàn
  chỉnh trong 1 chunk là trường hợp suy biến (1 fragment đầy đủ).
- `MockLLMClient.chat_stream` mới: consume script Y HỆT `chat()` (1 entry/turn,
  qua `_next_response` chung), yield text theo chunk `STREAM_CHUNK_SIZE=8`
  rồi `done` mang nguyên response (tool calls included) — cùng 1 script test
  được cả 2 chế độ.

### py.typed + mypy
- `kinetic_sdk/py.typed` (rỗng, PEP 561) + `[tool.setuptools.package-data]`
  trong pyproject — đã verify file nằm trong wheel build thật.
- `Tool.name/description/parameters` đổi từ `ClassVar[...]` sang instance
  annotation thuần: cả pattern class-attr (GitTool/TerminalTool/FileTool gán
  ở class level, bỏ `ClassVar` khỏi 3 trường này) lẫn pattern instance-attr
  (MockTool/MCPToolAdapter/DelegateTool gán trong `__init__`) đều hợp lệ.
  Trước đây base khai ClassVar → mypy "Cannot assign to class variable via
  instance" ở mọi tool động.
- GOTCHA mypy thật đã gặp: `def execute(self, command: str, **_: Any)` rồi
  `output, _ = proc.communicate()` — `_` trong `**_` là biến `dict[str, Any]`,
  unpack gán `str` vào `_` → lỗi tưởng như ở `output`. Fix: unpack thành
  `output, _err`.
- `_GuardedLLMClient.model` đổi từ read-only property sang plain instance
  attribute (property override writeable attr của `LLMClient` là lỗi mypy).
- `JSONLAuditLogger(path)` giờ nhận `str | os.PathLike[str]`.
- Các lỗi mypy khác đã fix thật: hostname Optional trong SSETransport (raise
  `MCPTransportError` khi URL không có host), `add_assistant` nhận
  `str | list[dict]`, plugin loader check `spec is None` trước
  `module_from_spec`, FileTool handler dict annotate `Callable[..., ToolResult]`.

### Điểm nhỏ
- `LICENSE` (MIT) ở root — khớp `license = "MIT"` trong pyproject.
- `kinetic_sdk/__init__.py` re-export tối thiểu: `Agent`, `AgentMode`,
  `Tool`, `ToolResult`, `PermissionPolicy`, `AllowListPolicy`,
  `PermissivePolicy`, `PermissionDecision`, `ConversationState`, `Event`,
  `EventBus`, `LLMClient`, `LLMResponse`, `ToolCall`, `StreamEvent` (+
  `__version__`). CỐ Ý không re-export hết — tránh circular import và rối
  API surface; mọi thứ khác import từ submodule như cũ.
- `tool_response(call_id=None, name="", arguments=None)`: `call_id` tự sinh
  `call-<uuid4>` khi bỏ trống (`tool_response(name="calc", arguments={...})`);
  thứ tự positional cũ `tool_response("id-1", "calc", {...})` vẫn chạy —
  KHÔNG đảo thứ tự tham số vì sẽ âm thầm phá mọi test hiện có. Thiếu `name`
  → `ValueError`.
