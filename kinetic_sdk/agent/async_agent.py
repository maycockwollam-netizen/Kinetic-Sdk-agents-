"""The async Kinetic agent: a fully non-blocking tool-calling loop.

:class:`AsyncAgent` is the async twin of
:class:`~kinetic_sdk.agent.agent.Agent`. It implements the SAME feature set
with the SAME semantics — FLASH/MAX routing and escalation, sticky MAX,
context compaction, hooks, permission policy + confirmation, audit,
observability, schema validation, cancellation, state persistence and
streaming — so behaviour is identical from the outside; only the mechanics
differ:

* The LLM is an :class:`~kinetic_sdk.llm.client.AsyncLLMClient` — every
  model turn is awaited, never blocking the event loop. A plain sync
  :class:`~kinetic_sdk.llm.client.LLMClient` is accepted too and wrapped in
  :class:`~kinetic_sdk.llm.async_client.SyncToAsyncLLMClient`
  automatically.
* Tools run through :meth:`Tool.execute_async` (default: the sync
  ``execute`` in a worker thread; natively async tools override it).
* Hooks run through :meth:`HookRegistry.trigger_async`, so coroutine hooks
  are awaited properly (the sync loop rejects them with a warning).
* Events are published via :meth:`EventBus.publish_async`, so coroutine
  subscribers are awaited instead of skipped.
* Parallel tool execution uses ``asyncio.gather`` instead of a thread pool,
  and a tool timeout cancels a native-async tool for real (a thread-bridged
  sync tool keeps running in the background — Python cannot kill threads).
* The classifier is an :class:`AsyncTaskClassifier`; a sync
  :class:`~kinetic_sdk.agent.classifier.TaskClassifier` is bridged with
  ``asyncio.to_thread``.

Concurrency note: like the sync ``Agent`` (and like ``EventBus`` itself),
one ``AsyncAgent`` instance is NOT re-entrant — do not drive two ``run()``
coroutines on the same instance concurrently. Create one agent per
concurrent task; independent agents run concurrently just fine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from kinetic_sdk.agent.async_classifier import (
    AsyncDefaultClassifier,
    AsyncTaskClassifier,
)
from kinetic_sdk.agent.budget import RunBudget, RunBudgetExceeded
from kinetic_sdk.agent.classifier import TaskClassifier
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.agent.structured import (
    CORRECTION_TEMPLATE,
    DEFAULT_STRUCTURED_RETRIES,
    check_final_answer,
    schema_instruction,
)
from kinetic_sdk.context.manager import (
    ContextManager,
    SimpleTruncateContextManager,
    SummarizingContextManager,
)
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import ConversationStore
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.hooks.registry import HookRegistry
from kinetic_sdk.llm.async_client import SyncToAsyncLLMClient
from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LLMClient,
    LLMResponse,
    ToolCall,
)
from kinetic_sdk.llm.usage import UsageAccumulator
from kinetic_sdk.memory.provider import MemoryProvider
from kinetic_sdk.observability.logger import ObservabilityLogger
from kinetic_sdk.security.audit import AuditLogger, InMemoryAuditLogger
from kinetic_sdk.security.policy import (
    AllowListPolicy,
    PermissionDecision,
    PermissionPolicy,
)
from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.tool.base import Tool, ToolResult
from kinetic_sdk.tool.validation import validate_tool_input

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Current UTC time (audit entries always use tz-aware ISO timestamps)."""
    return datetime.now(timezone.utc)


class _SyncClassifierBridge(AsyncTaskClassifier):
    """Bridge a synchronous :class:`TaskClassifier` into the async interface.

    The wrapped classifier's ``classify`` runs in a worker thread so a
    blocking (e.g. HTTP) classification call never stalls the event loop.
    The inner classifier's alias is preserved so log lines stay identical.
    """

    def __init__(self, inner: TaskClassifier) -> None:
        self.inner = inner
        self.alias = inner.alias

    async def classify(self, task: str):
        return await asyncio.to_thread(self.inner.classify, task)


class AsyncAgent:
    """A tool-calling agent driven by an :class:`AsyncLLMClient`.

    Args mirror :class:`~kinetic_sdk.agent.agent.Agent` — see that class for
    the full semantics of every option. Differences:

    * ``llm`` may be an :class:`AsyncLLMClient` OR a plain sync
      :class:`LLMClient` (auto-wrapped in
      :class:`~kinetic_sdk.llm.async_client.SyncToAsyncLLMClient`).
    * ``classifier`` may be an :class:`AsyncTaskClassifier` OR a sync
      :class:`~kinetic_sdk.agent.classifier.TaskClassifier` (auto-bridged
      via a worker thread). Default: :class:`AsyncDefaultClassifier`
      (always MAX), matching the sync agent's conservative default.
    * ``parallel_tool_execution=True`` fans a tool-call batch out with
      ``asyncio.gather`` — no thread pool involved for native-async tools.
    * ``tool_timeout`` uses ``asyncio.wait_for``: a timed-out NATIVE-async
      tool is cancelled for real; a thread-bridged sync tool is abandoned
      but keeps running in the background (documented, same as sync).

    Attributes:
        mode: Current :class:`AgentMode` (set once per run by the
            classifier; FLASH -> MAX escalation only, never the reverse).
        enable_extended_reasoning: Mode-driven flag (False in FLASH).
    """

    #: Per-mode iteration caps used when ``max_iterations`` is left to routing.
    MODE_MAX_ITERATIONS: dict[AgentMode, int] = {AgentMode.FLASH: 5, AgentMode.MAX: 50}

    #: When in FLASH, escalate to MAX after this many iterations without a
    #: final answer (must stay below the FLASH cap of 5).
    FLASH_ESCALATION_THRESHOLD: int = 3

    #: Default context-window size (tokens) assumed when the caller does not
    #: declare the model's real limit. Deliberately conservative.
    DEFAULT_MODEL_CONTEXT_LIMIT: int = 128_000

    def __init__(
        self,
        llm: AsyncLLMClient | LLMClient,
        tools: Iterable[Tool] | None = None,
        state: ConversationState | None = None,
        event_bus: EventBus | None = None,
        classifier: AsyncTaskClassifier | TaskClassifier | None = None,
        max_iterations: int | None = None,
        context_manager: ContextManager | None = None,
        model_context_limit: int | None = None,
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
        observability_logger: ObservabilityLogger | None = None,
        hooks: HookRegistry | None = None,
        tool_timeout: float | None = None,
        state_store: ConversationStore | None = None,
        validate_tool_inputs: bool = True,
        parallel_tool_execution: bool = False,
        memory: MemoryProvider | None = None,
        memory_recall_limit: int = 3,
        run_budget: RunBudget | None = None,
    ) -> None:
        if isinstance(llm, AsyncLLMClient):
            self.llm: AsyncLLMClient = llm
        elif isinstance(llm, LLMClient):
            self.llm = SyncToAsyncLLMClient(llm)
        else:
            raise TypeError(
                "llm must be an AsyncLLMClient or LLMClient, "
                f"got {type(llm).__name__}"
            )
        #: Optional persistence backend — same contract as the sync agent:
        #: resumed at construction when no explicit ``state`` is given and
        #: saved after every turn at replay-valid boundaries; store failures
        #: are logged, never fatal.
        self.state_store = state_store
        # NOTE: ``is not None``, not truthiness — ConversationState defines
        # __len__, so an empty state is falsy but still valid.
        if state is not None:
            self.state = state
        elif state_store is not None:
            resumed = state_store.load()
            self.state = resumed if resumed is not None else ConversationState()
        else:
            self.state = ConversationState()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        if classifier is None:
            self.classifier: AsyncTaskClassifier = AsyncDefaultClassifier()
        elif isinstance(classifier, AsyncTaskClassifier):
            self.classifier = classifier
        elif isinstance(classifier, TaskClassifier):
            self.classifier = _SyncClassifierBridge(classifier)
        else:
            raise TypeError(
                "classifier must be an AsyncTaskClassifier or TaskClassifier, "
                f"got {type(classifier).__name__}"
            )
        self.context_manager: ContextManager = (
            context_manager if context_manager is not None else SimpleTruncateContextManager()
        )
        if isinstance(self.context_manager, SummarizingContextManager) and (
            self.context_manager.event_bus is None
        ):
            # Route summarization-failure events through the agent's bus.
            self.context_manager.event_bus = self.event_bus
        self.model_context_limit: int = (
            model_context_limit if model_context_limit is not None else self.DEFAULT_MODEL_CONTEXT_LIMIT
        )
        self.permission_policy: PermissionPolicy = (
            permission_policy if permission_policy is not None else AllowListPolicy()
        )
        self.audit_logger: AuditLogger = (
            audit_logger if audit_logger is not None else InMemoryAuditLogger()
        )
        self.observability_logger = observability_logger
        if observability_logger is not None:
            # Attach at construction (not in run()) so run_started is caught.
            observability_logger.attach(self.event_bus)
        self.hooks = hooks
        if hooks is not None and hooks.event_bus is None:
            # Route hooks.error events through the agent's bus.
            hooks.event_bus = self.event_bus
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")
        #: Per-call wall-clock limit for ``Tool.execute_async``. ``None``
        #: disables the guard. On expiry the loop receives an error
        #: ToolResult and continues; a native-async tool is cancelled for
        #: real, a thread-bridged sync tool keeps running in the background.
        self.tool_timeout = tool_timeout
        #: When True, a batch of tool calls in one model turn is fanned out
        #: with ``asyncio.gather`` instead of running strictly in order.
        #: Gating (hooks, policy, confirmation, validation) still runs
        #: sequentially BEFORE anything executes; results are finalized
        #: (audit, AFTER hooks, state, events) in the model's original
        #: order. OPT-IN: tools sharing mutable state are not sound under
        #: concurrent execution.
        self.parallel_tool_execution = parallel_tool_execution
        #: Validate model-supplied arguments against each tool's declared
        #: JSON schema before executing (see ``tool/validation.py``).
        self.validate_tool_inputs = validate_tool_inputs
        #: Cooperative cancellation flag, set via :meth:`cancel` from any
        #: task. Checked between iterations and between tool calls.
        self._cancel_event = asyncio.Event()
        #: UUID of the in-flight (or most recent) :meth:`run`; ``None``
        #: before the first run. Every event emitted during a run carries it.
        self._run_id: str | None = None
        # User override of the iteration cap. ``None`` => let routing pick
        # per mode. Stored separately so an escalation can re-derive the cap.
        self._max_iterations_override: int | None = max_iterations
        self.max_iterations: int = max_iterations if max_iterations is not None else self.MODE_MAX_ITERATIONS[AgentMode.MAX]
        self.mode: AgentMode = AgentMode.MAX
        self.enable_extended_reasoning: bool = True
        #: Once any run of this conversation classified MAX (or escalated to
        #: it), later runs never route back to FLASH. Set on successful MAX
        #: classification and on escalation - NOT on the exception fallback.
        self._sticky_max: bool = False
        #: Running LLM usage totals across all runs (per-run data is in the
        #: ``llm.usage`` events; see ``llm/usage.py``).
        self.usage = UsageAccumulator()
        #: Parsed final answer of the last ``run(output_schema=...)`` (None
        #: when no schema was requested or validation ultimately failed).
        self.structured_output: Any = None
        #: Optional long-term memory (recall before runs, store after).
        self.memory = memory
        self.memory_recall_limit = memory_recall_limit
        #: Optional root-run cost guardrail — same contract as the sync
        #: Agent: checked before every LLM call; exhaustion stops the run
        #: gracefully with ``agent.budget_exceeded`` and an explanatory
        #: final text. ``None`` (default) = unlimited.
        self.run_budget = run_budget

        tool_list = list(tools or [])
        self._tools: dict[str, Tool] = {}
        for tool in tool_list:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    # --- public API ---------------------------------------------------

    def add_tool(self, tool: Tool) -> None:
        """Register an additional tool at runtime.

        Raises if a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return the tool definitions to send to the model."""
        return [t.to_schema() for t in self._tools.values()]

    async def run(
        self,
        user_message: str | None = None,
        *,
        stream: bool = False,
        output_schema: dict[str, Any] | None = None,
        structured_retries: int | None = None,
    ) -> str:
        """Run the agent loop until the model stops calling tools.

        Semantics are identical to :meth:`Agent.run`: the task is classified
        exactly once before the first turn (FLASH/MAX routing, sticky MAX,
        mid-run FLASH -> MAX escalation on first-turn tool errors or after
        :attr:`FLASH_ESCALATION_THRESHOLD` unanswered iterations), every turn
        emits the same events, and the final assistant text is returned.

        Args:
            user_message: Optional user turn to append before running. Pass
                ``None`` to continue an existing conversation.
            stream: When ``True``, LLM turns use ``chat_stream()`` and every
                text chunk is published in real time as an
                ``agent.text_delta`` event (payload: ``{"delta": str}``).
                Clients without streaming support fall back to a plain
                ``chat()`` call (logged, no deltas emitted).

        Returns:
            The final assistant text. On ``max_iterations`` the last text
            seen (possibly empty) is returned and ``agent.error`` is
            published; on :meth:`cancel` the best text so far is returned
            and ``agent.cancelled`` is published.
        """
        self.structured_output = None
        if user_message is not None and self.memory is not None:
            await self._recall_memory(user_message)
        composed = user_message
        if output_schema is not None:
            composed = (user_message or "") + "\n\n" + schema_instruction(output_schema)
        if composed is not None:
            self.state.add_user_message(composed)

        self._cancel_event.clear()  # a new run starts un-cancelled
        self._run_id = str(uuid.uuid4())
        await self._trigger_hooks(
            HookPoint.BEFORE_RUN,
            HookContext(
                point=HookPoint.BEFORE_RUN, run_id=self._run_id, user_message=user_message
            ),
        )
        await self._classify_and_route(user_message)

        await self._emit(
            "agent.run_started",
            {"mode": self.mode.value, "tools": list(self._tools), "max_iterations": self.max_iterations},
        )

        final_text = ""
        try:
            final_text = await self._run_loop(
                stream=stream,
                output_schema=output_schema,
                structured_retries=structured_retries,
            )
        except Exception as exc:
            await self._trigger_hooks(
                HookPoint.ON_ERROR,
                HookContext(
                    point=HookPoint.ON_ERROR,
                    run_id=self._run_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            await self._emit("agent.error", {"reason": "exception", "error": str(exc)})
            raise

        await self._trigger_hooks(
            HookPoint.AFTER_RUN,
            HookContext(
                point=HookPoint.AFTER_RUN, run_id=self._run_id, final_text=final_text
            ),
        )
        await self._emit("agent.run_finished", {"final_text": final_text, "mode": self.mode.value})
        if user_message is not None and self.memory is not None:
            await self._store_memory(user_message, final_text)
        return final_text

    def escalate(self) -> bool:
        """Escalate from FLASH to MAX mid-task. Returns True if it escalated.

        Synchronous like the sync agent's — it only mutates local state.
        Unlike the sync version the ``agent.escalated`` event is NOT emitted
        here (event publishing is async); the loop emits it right after a
        successful escalation, so observers see the exact same event stream.
        """
        target = AgentMode.escalates_to(self.mode)
        if target is None:
            return False
        self.mode = target
        self.enable_extended_reasoning = target is AgentMode.MAX
        if target is AgentMode.MAX:
            # Once a conversation has run at MAX it never drops back to
            # FLASH - a mid-conversation downgrade would silently shrink the
            # iteration budget of later turns.
            self._sticky_max = True
        if self._max_iterations_override is None:
            self.max_iterations = self.MODE_MAX_ITERATIONS[target]
        return True

    def cancel(self) -> None:
        """Ask the in-flight :meth:`run` to stop cooperatively.

        Synchronous and safe to call from any task or callback. The loop
        notices between iterations and between tool calls, fills in error
        results for any skipped tool calls (so the history stays
        replay-valid), emits ``agent.cancelled`` and returns the best final
        text it has. A tool call already executing is NOT interrupted —
        cooperative cancellation cannot preempt running work.
        """
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        """True after :meth:`cancel` until the next :meth:`run` starts."""
        return self._cancel_event.is_set()

    @property
    def run_id(self) -> str | None:
        """UUID of the current/last :meth:`run`, or ``None`` before any run."""
        return self._run_id

    # --- internals ----------------------------------------------------

    async def _classify_and_route(self, user_message: str | None) -> None:
        """Classify the task once and apply the mode-specific config."""
        task = user_message if user_message is not None else self._first_user_message() or ""
        try:
            classification = await self.classifier.classify(task)
        except Exception as exc:  # noqa: BLE001 - safe-side fallback
            logger.warning(
                "Classifier %s raised (routing to MAX): %s",
                self.classifier.alias,
                type(exc).__name__,
            )
            classification = None
        if classification is None:
            self._apply_mode(AgentMode.MAX, rationale="classifier_exception")
            return
        rationale = classification.rationale
        mode = classification.mode
        if mode is AgentMode.FLASH and self._sticky_max:
            # MAX is sticky across runs of one conversation: a run that
            # classified MAX (or escalated) must not be followed by a FLASH
            # run with its much smaller iteration budget.
            mode = AgentMode.MAX
            rationale = (rationale + " " if rationale else "") + "[max sticky: downgrade prevented]"
        elif mode is AgentMode.MAX:
            self._sticky_max = True
        self._apply_mode(mode, rationale=rationale)
        await self._emit(
            "agent.classified",
            {
                "complexity": classification.complexity.value,
                "mode": classification.mode.value,
                "applied_mode": mode.value,
                "confidence": classification.confidence,
                "rationale": rationale,
            },
        )

    def _apply_mode(self, mode: AgentMode, rationale: str = "") -> None:
        """Set :attr:`mode` and derive the iteration cap + reasoning flag."""
        self.mode = mode
        self.enable_extended_reasoning = mode is AgentMode.MAX
        if self._max_iterations_override is not None:
            self.max_iterations = self._max_iterations_override
        else:
            self.max_iterations = self.MODE_MAX_ITERATIONS[mode]

    async def _run_loop(
        self,
        stream: bool = False,
        output_schema: dict[str, Any] | None = None,
        structured_retries: int | None = None,
    ) -> str:
        """The tool-calling loop, with mid-run FLASH -> MAX escalation."""
        final_text = ""
        iteration = 0
        escalated = False
        retries_left = (
            DEFAULT_STRUCTURED_RETRIES if structured_retries is None else structured_retries
        )
        while iteration < self.max_iterations:
            if self._cancel_event.is_set():
                logger.info("Agent run cancelled at iteration %d", iteration)
                await self._emit("agent.cancelled", {"iteration": iteration})
                return final_text
            await self._emit("agent.turn_started", {"iteration": iteration, "mode": self.mode.value})
            await self._maybe_compact_context()
            await self._trigger_hooks(
                HookPoint.BEFORE_LLM_CALL,
                HookContext(
                    point=HookPoint.BEFORE_LLM_CALL,
                    run_id=self._run_id,
                    iteration=iteration,
                ),
            )
            # Re-check after hooks: a hook is a common place for UI-driven
            # cancellation, and it should prevent the imminent LLM call.
            if self._cancel_event.is_set():
                logger.info("Agent run cancelled at iteration %d", iteration)
                await self._emit("agent.cancelled", {"iteration": iteration})
                return final_text
            if self.run_budget is not None:
                try:
                    self.run_budget.check()
                except RunBudgetExceeded as exc:
                    logger.warning("Run budget exceeded: %s", exc)
                    await self._emit(
                        "agent.budget_exceeded", self.run_budget.details()
                    )
                    return f"Run stopped: budget exceeded ({exc})."
            response = await self._call_llm(stream=stream)
            await self._trigger_hooks(
                HookPoint.AFTER_LLM_CALL,
                HookContext(
                    point=HookPoint.AFTER_LLM_CALL,
                    run_id=self._run_id,
                    iteration=iteration,
                    llm_response=response,
                ),
            )
            self.state.add_assistant(self._assistant_content(response))
            await self._emit(
                "agent.llm_response",
                {"tool_calls": len(response.tool_calls), "stop_reason": response.stop_reason},
            )

            if not response.tool_calls:
                final_text = response.content
                if output_schema is not None:
                    decision = await self._check_structured(
                        output_schema, final_text, retries_left
                    )
                    if decision == "retry":
                        retries_left -= 1
                        iteration += 1
                        continue
                # Persist only at replay-valid boundaries: an assistant text
                # turn (no dangling tool_use) or right after tool results.
                await self._persist_state()
                break

            any_error = await self._execute_tool_calls(response.tool_calls)
            await self._persist_state()

            if not escalated and self.mode is AgentMode.FLASH:
                should_escalate = False
                if iteration == 0 and any_error:
                    should_escalate = True
                elif (iteration + 1) >= self.FLASH_ESCALATION_THRESHOLD:
                    should_escalate = True
                if should_escalate:
                    previous = self.mode
                    escalated = self.escalate()
                    if escalated:
                        await self._emit(
                            "agent.escalated",
                            {"from": previous.value, "to": self.mode.value},
                        )
            iteration += 1
        else:
            logger.warning("Agent hit max_iterations=%d", self.max_iterations)
            await self._emit(
                "agent.error",
                {"reason": "max_iterations", "iterations": self.max_iterations},
            )
        return final_text

    def _first_user_message(self) -> str | None:
        """Return the most recent user text message, if any (for re-runs)."""
        for msg in reversed(self.state.messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None

    async def _maybe_compact_context(self) -> None:
        """Compact the history before an LLM call when over the threshold.

        ``should_compact``/``compact`` are synchronous and CPU-bound (token
        estimation + message copying); a ``SummarizingContextManager`` may
        make a blocking LLM call inside ``compact``, so the whole thing runs
        in a worker thread to keep the event loop responsive.
        """
        manager = self.context_manager
        if manager is None:
            return
        should = await asyncio.to_thread(
            manager.should_compact, self.state, self.model_context_limit
        )
        if not should:
            return
        before = len(self.state.messages)
        self.state = await asyncio.to_thread(manager.compact, self.state)
        after = len(self.state.messages)
        logger.info("Context compacted: %d -> %d messages", before, after)
        await self._emit(
            "context.compacted",
            {
                "manager": type(manager).__name__,
                "messages_before": before,
                "messages_after": after,
                "messages_removed": before - after,
                "estimated_tokens_after": manager.estimate_state_tokens(self.state),
            },
        )

    async def _call_llm(self, stream: bool = False) -> LLMResponse:
        """Ask the LLM for the next turn using the current history + tools.

        With ``stream=True`` the client's ``chat_stream`` is consumed:
        each text chunk is re-published as an ``agent.text_delta`` event as
        it arrives, and the aggregated response from the final ``done``
        event is returned, so the rest of the loop (tool calling,
        escalation) works unchanged. A client that does not implement
        streaming falls back to a plain ``chat`` call.
        """
        system, messages = self.state.for_llm()
        tools = self.tool_schemas() or None
        if not stream:
            response = await self.llm.chat(messages=messages, tools=tools, system=system)
            await self._record_usage(response)
            return response
        try:
            final: LLMResponse | None = None
            async for event in self.llm.chat_stream(messages=messages, tools=tools, system=system):
                if event.type == "text" and isinstance(event.delta, str):
                    await self._emit("agent.text_delta", {"delta": event.delta})
                elif event.type == "done" and isinstance(event.delta, LLMResponse):
                    final = event.delta
        except NotImplementedError:
            logger.warning(
                "%s does not support streaming; falling back to chat()",
                type(self.llm).__name__,
            )
            response = await self.llm.chat(messages=messages, tools=tools, system=system)
            await self._record_usage(response)
            return response
        if final is None:
            # Stream ended without a done event: treat as an empty end turn
            # rather than crashing the loop.
            final = LLMResponse(content="", stop_reason="end_turn")
        await self._record_usage(final)
        return final

    async def _check_structured(
        self, schema: dict[str, Any], text: str, retries_left: int
    ) -> str:
        """Validate a final answer against *schema*; return "break" | "retry".

        Mirrors the sync agent's check: success sets ``structured_output``
        and emits ``agent.structured_output_parsed``; a failure with rounds
        left appends a correction user turn and emits
        ``agent.structured_output_retry``; the exhausted case emits
        ``agent.structured_output_invalid`` and keeps the raw text.
        """
        ok, value, problems = check_final_answer(schema, text)
        if ok:
            self.structured_output = value
            await self._emit("agent.structured_output_parsed", {"schema": schema})
            return "break"
        if retries_left > 0:
            self.state.add_user_message(
                CORRECTION_TEMPLATE.format(problems="; ".join(problems))
            )
            await self._emit("agent.structured_output_retry", {"problems": problems})
            return "retry"
        await self._emit("agent.structured_output_invalid", {"problems": problems})
        return "break"

    async def _recall_memory(self, query: str) -> None:
        """Inject relevant memories before the task (fail-soft, like sync)."""
        assert self.memory is not None
        try:
            recalled = self.memory.search(query, limit=self.memory_recall_limit)
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("Memory recall failed: %s", exc)
            await self._emit(
                "agent.memory_recall_failed",
                {"error": redact_secrets(f"{type(exc).__name__}: {exc}")},
            )
            return
        if not recalled:
            return
        lines = "\n".join("- " + e.text for e in recalled)
        self.state.add_user_message(f"[Recalled memories]\n{lines}")
        await self._emit(
            "agent.memory_recalled", {"query": query, "count": len(recalled)}
        )

    async def _store_memory(self, user_message: str, final_text: str) -> None:
        """Remember the completed Q/A pair (fail-soft, like sync)."""
        assert self.memory is not None
        try:
            entry = self.memory.add(
                f"User: {user_message}\nAssistant: {final_text}",
                metadata={"run_id": self._run_id},
            )
            await self._emit("agent.memory_stored", {"id": entry.id})
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("Memory store failed: %s", exc)
            await self._emit(
                "agent.memory_store_failed",
                {"error": redact_secrets(f"{type(exc).__name__}: {exc}")},
            )

    async def _record_usage(self, response: LLMResponse) -> None:
        """Fold one LLM turn's usage into the agent total + emit a delta."""
        if self.run_budget is not None:
            # Count the call even when the provider reports no usage, so a
            # call-limited budget cannot be bypassed by a usage-silent client.
            self.run_budget.record(response.usage)
        if not response.usage:
            return
        self.usage.record(response.usage)
        await self._emit("llm.usage", {"usage": dict(response.usage)})

    async def _persist_state(self) -> None:
        """Save the conversation to the configured store, never fatally.

        Only called at replay-valid boundaries (assistant text turn or
        right after tool results), so a resumed history never ends on a
        dangling ``tool_use``. The save itself runs in a worker thread —
        store I/O must not stall the event loop.
        """
        if self.state_store is None:
            return
        try:
            await asyncio.to_thread(self.state_store.save, self.state)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to persist conversation state: %s", exc)
            await self._emit(
                "agent.state_persist_failed",
                {"error": redact_secrets(f"{type(exc).__name__}: {exc}")},
            )

    def _assistant_content(self, response: LLMResponse) -> Any:
        """Build the assistant message ``content`` to store in history.

        Reproduces Anthropic's content-block shape so the history can be
        replayed to the model verbatim: text blocks + tool_use blocks.
        """
        blocks: list[dict[str, Any]] = []
        if response.content:
            blocks.append({"type": "text", "text": response.content})
        for call in response.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return blocks if blocks else response.content

    async def _execute_tool_calls(self, calls: list[ToolCall]) -> bool:
        """Execute each tool call in order and append results to the state.

        Returns ``True`` if at least one tool call resulted in an error
        (used by the escalation logic to detect a failing first turn).

        When the run is cancelled mid-batch the remaining calls are NOT
        executed, but each still gets an error ``tool_result`` appended —
        every ``tool_use`` in the history must have its result or providers
        reject the whole conversation on the next call.
        """
        if self.parallel_tool_execution and len(calls) > 1:
            return await self._execute_tool_calls_parallel(calls)
        any_error = False
        for call in calls:
            if self._cancel_event.is_set():
                result = ToolResult(error="skipped: run cancelled")
                self.state.add_tool_result(
                    call.id, self._format_tool_output(result), is_error=True
                )
                any_error = True
                continue
            await self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            result = await self._execute_one(call)
            any_error = any_error or result.is_error
            await self._emit_tool_call_finished(call, result)
            self.state.add_tool_result(
                call.id,
                self._format_tool_output(result),
                is_error=result.is_error,
            )
        return any_error

    async def _execute_tool_calls_parallel(self, calls: list[ToolCall]) -> bool:
        """Fan a multi-call batch out with ``asyncio.gather``.

        Ordering guarantees, identical to the sequential path from the
        outside: gating happens per call in order, results are finalized
        and appended to the state in the model's original order, and every
        event is emitted from the loop task. Only ``Tool.execute_async``
        itself runs concurrently — a cancelled run still cannot preempt
        in-flight executions, and skipped calls still receive their error
        ``tool_result`` so the history stays valid.
        """
        any_error = False
        early_or_skip: dict[int, ToolResult] = {}
        skipped: set[int] = set()
        pending: list[tuple[int, ToolCall, Tool, dict[str, Any]]] = []

        for i, call in enumerate(calls):
            if self._cancel_event.is_set():
                early_or_skip[i] = ToolResult(error="skipped: run cancelled")
                skipped.add(i)
                continue
            await self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            tool, arguments, early = await self._prepare_call(call)
            if early is not None:
                early_or_skip[i] = early
            else:
                assert tool is not None
                pending.append((i, call, tool, arguments))

        executed: dict[int, ToolResult] = {}
        executed_args: dict[int, dict[str, Any]] = {}
        if pending:
            async def _run(i: int, call: ToolCall, tool: Tool, arguments: dict[str, Any]) -> None:
                executed_args[i] = arguments
                try:
                    executed[i] = await self._run_tool(tool, arguments)
                except Exception as exc:  # noqa: BLE001 - surface as tool error
                    logger.exception("Tool %s raised", call.name)
                    executed[i] = ToolResult(error=f"{type(exc).__name__}: {exc}")

            await asyncio.gather(
                *(_run(i, call, tool, arguments) for (i, call, tool, arguments) in pending)
            )

        for i, call in enumerate(calls):
            if i in early_or_skip:
                result = early_or_skip[i]
                any_error = any_error or result.is_error
                if i not in skipped:
                    await self._emit_tool_call_finished(call, result)
                self.state.add_tool_result(
                    call.id, self._format_tool_output(result), is_error=result.is_error
                )
                continue
            result = executed[i]
            any_error = any_error or result.is_error
            await self._finalize_call(call, executed_args[i], result)
            await self._emit_tool_call_finished(call, result)
            self.state.add_tool_result(
                call.id, self._format_tool_output(result), is_error=result.is_error
            )
        return any_error

    async def _execute_one(self, call: ToolCall) -> ToolResult:
        """Dispatch a single tool call, gated by hooks + permission policy.

        Same flow as the sync agent's ``_execute_one``: BEFORE_TOOL_CALL
        hooks (cancel / replace input) -> permission policy + audit ->
        confirmation via ON_PERMISSION_CHECK hooks -> unknown-tool and
        schema validation -> execute -> audit + AFTER_TOOL_CALL hooks.
        Denied/cancelled calls return an error :class:`ToolResult` so the
        model can react, and denials emit ``security.permission_denied``.
        """
        tool, arguments, early = await self._prepare_call(call)
        if early is not None:
            return early
        assert tool is not None  # guaranteed by _prepare_call's contract
        try:
            result = await self._run_tool(tool, arguments)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("Tool %s raised", call.name)
            result = ToolResult(error=f"{type(exc).__name__}: {exc}")
        await self._finalize_call(call, arguments, result)
        return result

    async def _prepare_call(
        self, call: ToolCall
    ) -> tuple[Tool | None, dict[str, Any], ToolResult | None]:
        """Run pre-execution gating for one call, sequentially.

        Steps 1-4 of :meth:`_execute_one`: BEFORE_TOOL_CALL hooks,
        permission policy + audit, confirmation fallback, unknown-tool
        check, and input schema validation. Returns ``(tool, arguments,
        None)`` when the call may execute, or ``(None, arguments,
        error_result)`` when it was short-circuited. Contains no concurrent
        work: safe to run for a whole batch before executions are fanned
        out with ``asyncio.gather``.
        """
        arguments = call.arguments
        if self.hooks is not None:
            results = await self._trigger_hooks(
                HookPoint.BEFORE_TOOL_CALL,
                HookContext(
                    point=HookPoint.BEFORE_TOOL_CALL,
                    run_id=self._run_id,
                    tool_name=call.name,
                    tool_input=arguments,
                ),
            )
            if any(not r.should_continue for r in results):
                return None, arguments, ToolResult(
                    error=f"Tool call {call.name!r} cancelled by a before_tool_call hook"
                )
            for r in results:
                if r.modified_context and isinstance(
                    r.modified_context.get("tool_input"), dict
                ):
                    arguments = r.modified_context["tool_input"]

        decision = self.permission_policy.check(call.name, arguments)
        self.audit_logger.log_tool_call(call.name, arguments, decision, _utcnow())
        if not decision.allowed:
            return None, arguments, await self._deny_call(call, decision.reason, arguments)
        if decision.requires_confirmation and not await self._confirmed_by_hooks(
            call, arguments, decision
        ):
            return None, arguments, await self._deny_call(
                call,
                "requires manual confirmation, not yet supported in automated "
                f"mode ({decision.reason})",
                arguments,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            logger.error("Unknown tool requested: %s", call.name)
            return None, arguments, ToolResult(error=f"Unknown tool: {call.name}")
        if self.validate_tool_inputs:
            problems = validate_tool_input(tool.parameters, arguments)
            if problems:
                result = ToolResult(error="invalid tool input: " + "; ".join(problems))
                self.audit_logger.log_tool_result(call.name, result, _utcnow())
                return None, arguments, result
        return tool, arguments, None

    async def _finalize_call(
        self, call: ToolCall, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """Audit + AFTER_TOOL_CALL hooks for an EXECUTED call."""
        self.audit_logger.log_tool_result(call.name, result, _utcnow())
        await self._trigger_hooks(
            HookPoint.AFTER_TOOL_CALL,
            HookContext(
                point=HookPoint.AFTER_TOOL_CALL,
                run_id=self._run_id,
                tool_name=call.name,
                tool_input=arguments,
                tool_result=result,
            ),
        )

    async def _run_tool(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        """Run ``tool.execute_async``, optionally under a wall-clock timeout.

        With ``tool_timeout`` set the call is wrapped in
        ``asyncio.wait_for``: on expiry a native-async tool is cancelled for
        real and the loop receives an error result; a thread-bridged sync
        tool (the default ``execute_async``) is abandoned but keeps running
        in the background — Python cannot safely kill a running thread, so
        tools wrapping subprocesses/network should enforce their own real
        timeout too (``GitTool`` already does).
        """
        if self.tool_timeout is None:
            return await tool.execute_async(**arguments)
        try:
            return await asyncio.wait_for(
                tool.execute_async(**arguments), timeout=self.tool_timeout
            )
        except asyncio.TimeoutError:
            return ToolResult(
                error=(
                    f"Tool {tool.name!r} timed out after {self.tool_timeout}s "
                    "(the call may still be running in the background)"
                )
            )

    async def _trigger_hooks(
        self, point: HookPoint, context: HookContext
    ) -> list[HookResult]:
        """Run the hooks registered for *point*, or nothing when unconfigured."""
        if self.hooks is None:
            return []
        return await self.hooks.trigger_async(point, context)

    async def _confirmed_by_hooks(
        self, call: ToolCall, arguments: dict[str, Any], decision: PermissionDecision
    ) -> bool:
        """Ask ``ON_PERMISSION_CHECK`` hooks to confirm a flagged tool call.

        Returns ``True`` when at least one hook explicitly allows the call
        (``should_continue=True``). No hooks configured, none registered at
        this point, or all declining (``False``/``None``) means *not*
        confirmed — the safe default.
        """
        if self.hooks is None:
            return False
        results = await self._trigger_hooks(
            HookPoint.ON_PERMISSION_CHECK,
            HookContext(
                point=HookPoint.ON_PERMISSION_CHECK,
                run_id=self._run_id,
                tool_name=call.name,
                tool_input=arguments,
                permission_decision=decision,
            ),
        )
        return any(r.should_continue for r in results)

    async def _deny_call(
        self, call: ToolCall, reason: str, tool_input: dict[str, Any] | None = None
    ) -> ToolResult:
        """Audit-log + emit a denial and build the error result for the model."""
        tool_input = tool_input if tool_input is not None else call.arguments
        logger.warning("Permission denied for tool %s: %s", call.name, reason)
        self.audit_logger.log_permission_denied(call.name, tool_input, reason, _utcnow())
        await self._emit(
            "security.permission_denied",
            {
                "name": call.name,
                "id": call.id,
                "reason": reason,
                "input_preview": redact_secrets(self._preview(tool_input)),
            },
        )
        return ToolResult(error=f"Permission denied for tool {call.name!r}: {reason}")

    async def _emit_tool_call_finished(self, call: ToolCall, result: ToolResult) -> None:
        await self._emit(
            "agent.tool_call_finished",
            {
                "name": call.name,
                "id": call.id,
                "is_error": result.is_error,
                "output_preview": redact_secrets(self._preview(result.output)),
            },
        )

    @staticmethod
    def _format_tool_output(result: ToolResult) -> str:
        """Serialise a ToolResult to a string the model can read back."""
        if result.is_error:
            return json.dumps({"error": result.error})
        payload = result.output
        try:
            return json.dumps(payload) if not isinstance(payload, str) else payload
        except (TypeError, ValueError):
            return str(payload)

    @staticmethod
    def _preview(value: Any, limit: int = 200) -> str:
        """Truncate a value to a short string for event payloads/logs."""
        s = str(value)
        return s if len(s) <= limit else s[:limit] + "..."

    async def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a lifecycle event if a bus is attached.

        Uses :meth:`EventBus.publish_async` so coroutine subscribers are
        awaited (the sync agent's ``publish`` would skip them). When a run
        is in flight its ``run_id`` is added to the payload so
        observability subscribers can group events per run.
        """
        if self.event_bus is None:
            return
        if self._run_id is not None:
            payload = {**payload, "run_id": self._run_id}
        await self.event_bus.publish_async(
            Event(type=event_type, payload=payload, source="agent")
        )
