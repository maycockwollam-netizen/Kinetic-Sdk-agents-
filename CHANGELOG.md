# Changelog

All notable changes to `kinetic-agent-sdk`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed

- **Parallel sub-agent safety** — shared event, audit, observability and
  metrics sinks now synchronise concurrent writes. Multiple `delegate` tool
  calls in one parallel tool batch can safely run sub-agents at the same time
  without racing subscribers, losing metric updates, or corrupting JSONL
  audit records.

## [0.2.0] — 2026-08-24

Stage 5 — Production readiness: closes the main feature gaps against the
mature agent SDKs (OpenHands, OpenAI Agents, LangChain/LangGraph, Pydantic
AI, AutoGen, Google ADK, Claude Agent SDK).

### Added

- **Structured output** — `Agent.run(..., output_schema=schema)` (sync +
  async): the final answer is parsed as JSON and validated against the
  schema (`tool/validation.py` subset); failed answers become correction
  turns (`structured_retries`, default 2). Parsed value on
  `agent.structured_output`; events `agent.structured_output_parsed` /
  `_retry` / `_invalid`. New helper module `agent/structured.py` and the
  public `validate_value()` in `tool/validation.py`.
- **Usage / cost tracking** — `llm/usage.py` (`UsageAccumulator`,
  `UsageSnapshot`); `agent.usage` totals across runs, one `llm.usage`
  event per LLM turn, `RunTrace.usage()` aggregated into
  `to_summary()["usage"]`. `LiteLLMClient` fills `usage["cost_usd"]`
  best-effort via `litellm.completion_cost` (sync + async clients).
- **OpenTelemetry export** — `observability/otel.py`
  (`OTelObservabilityLogger`): runs become `kinetic.run` spans, events
  become span events; optional extra `pip install kinetic-agent-sdk[otel]`.
- **Metrics** — `observability/metrics.py` (`MetricsCollector`): event
  counts, tool calls/failures, token totals, run durations via
  `snapshot()`.
- **Ask-user tool** — `ask_user/` (`AskUserTool`): the agent asks the
  operator a question mid-run through an injectable handler (CLI
  `input()` fallback).
- **Long-term memory** — `memory/` package: `MemoryProvider` ABC,
  `InMemoryMemory` + `JsonFileMemory` (keyword-overlap ranking),
  `MemoryTool` (store/search/list/clear). `Agent(memory=...)` auto-recalls
  before a run (`agent.memory_recalled`) and stores the Q/A after it
  (`agent.memory_stored`).
- **Agent server** — `server/` (`AgentServer`): stdlib-only REST API
  (`GET /health`, `POST /runs`, `GET /runs/<id>`) with optional bearer
  token auth and a 1 MB body cap.
- **Checkpointing** — `conversation/checkpoint.py`: `fork_store` (branch a
  saved conversation) and `rewind_state` (truncate to a replay-valid
  boundary; dangling `tool_use` cuts raise `CheckpointError`).
- **Eval harness** — `eval/` package: `EvalCase`, callable scorers
  (`exact_match`, `contains_expected`, `no_tool_failures`,
  `tool_called(name)`), and `EvalRunner` producing aggregate reports with
  per-case `RunTrace`.
- **Docker sandbox adapters** — `terminal/docker.py`
  (`docker_exec_wrapper`, `docker_run_wrapper`) + `TerminalTool(
  command_wrapper=...)`: commands execute as `<wrapper> bash -c <cmd>`
  inside a container; the tool's timeout/process-group kill still applies.
- **Async sub-agents** — `subagent/async_delegation.py` +
  `subagent/async_tool.py`: `spawn_async_subagent` / `run_async_subagent`
  / `AsyncDelegateTool` mirroring the sync path, including the guarded
  LLM client (budget + circuit breaker) and shared-tree budget.
- **MCP reconnect** — `MCPServerRegistry.reconnect(name)`: close the
  cached client and re-handshake (heals dead pipes on server restart).
- **Examples** — `examples/01_quickstart.py`,
  `02_server_and_structured_output.py`, `03_eval_and_memory.py` (all
  offline-runnable).

### Changed

- `LLMResponse.usage` values are now `Any`-typed (token counts stay ints;
  `cost_usd` is a float).
- `RunTrace.to_summary()` includes a `usage` entry.
- `Agent` / `AsyncAgent` gained `memory`, `memory_recall_limit` and
  `output_schema` / `structured_retries` (on `run()`).

## [0.1.0] — 2026-08-21

Stages 1-4 + hardening + async loop: core tool-calling loop, FLASH/MAX
routing, context compaction, secret management, security/observability/
hooks/testing, git/workspace/profiles/MCP/skills/plugin/subagent,
persistence, cancellation, streaming, full async layer.
