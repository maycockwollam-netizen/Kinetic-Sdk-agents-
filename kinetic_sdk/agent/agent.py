"""The Kinetic agent: a tool-calling loop.

This module wires together the Stage 1 building blocks into the core agent
loop:

1. Read the conversation history from :class:`ConversationState`.
2. Ask the :class:`LLMClient` for the next turn (with the available tools).
3. If the model requested tool calls, execute each registered :class:`Tool`
   and append the results back to the conversation, then loop.
4. If the model produced a final text answer, return it.

Stage 1 implements the loop; Stage 2 wires in FLASH/MAX routing
(``classifier.py``) and context-window compaction (``context/manager.py``):
before each LLM call the configured :class:`ContextManager` may replace the
history with a reduced copy, emitting ``context.compacted``. The loop is
synchronous and deterministic, which keeps tests simple.

The loop emits events on an optional :class:`EventBus` so observers can react
to each step without coupling to the agent internals.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any, Iterable

from kinetic_sdk.agent.budget import RunBudget, RunBudgetExceeded
from kinetic_sdk.agent.classifier import DefaultClassifier, TaskClassifier
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.agent.structured import (
    CORRECTION_TEMPLATE,
    DEFAULT_STRUCTURED_RETRIES,
    check_final_answer,
    schema_instruction,
)
from kinetic_sdk.agent.stuck_detector import StuckDetector
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
from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
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


class Agent:
    """A tool-calling agent bound to one LLM and a set of tools.

    Args:
        llm: The model client used for every reasoning turn.
        tools: Tools the agent may call. Duplicated tool names raise on
            construction to keep the dispatch table unambiguous.
        state: Conversation state. A fresh one is created if omitted.
        event_bus: Optional event bus the loop publishes lifecycle events to.
            Events emitted (see module docstring for the full list):
              ``agent.run_started``, ``agent.turn_started``,
              ``agent.llm_response``, ``agent.text_delta`` (only when
              ``run(stream=True)``), ``agent.tool_call_started``,
              ``agent.tool_call_finished``, ``agent.run_finished``,
              ``agent.escalated``, ``agent.classified``, ``agent.error``,
              ``agent.budget_exceeded``, ``context.compacted``,
              ``context.summarization_failed``, ``security.permission_denied``,
              ``agent.stuck_detected``, ``hooks.error`` (via the hook registry).
        classifier: Optional :class:`TaskClassifier`. When provided (or when
            the default is used) :meth:`run` classifies the task exactly once
            before the first turn and routes to FLASH or MAX. Pass ``None`` to
            fall back to :class:`DefaultClassifier` (always MAX).
        max_iterations: Safety cap on LLM turns per :meth:`run` to prevent
            infinite tool-calling loops. When ``None`` (the default) the cap is
            chosen by the routed mode: FLASH -> 5, MAX -> 50. An explicit value
            overrides the mode default for the *initial* mode; an escalation
            FLASH -> MAX mid-task still raises the cap to the MAX default.
        context_manager: Strategy that keeps the history inside the model's
            context window. ``None`` (default) uses
            :class:`SimpleTruncateContextManager`; pass
            :class:`NoopContextManager` to disable compaction entirely, or
            :class:`SummarizingContextManager` to replace elided spans with an
            LLM-generated summary (the agent's bus is wired into it so
            ``context.summarization_failed`` is observable).
        model_context_limit: The model's context window in tokens, used as
            the reference for the manager's safety threshold.
        permission_policy: Gate checked before every tool execution. ``None``
            (default) uses an empty :class:`AllowListPolicy` — deny-by-default,
            so SDK users must explicitly declare which tools may run. Pass
            :class:`PermissivePolicy` only for local dev/test.
        audit_logger: Sink recording every tool call, denial and result.
            ``None`` (default) uses :class:`InMemoryAuditLogger`.
        observability_logger: Optional structured event logger. ``None``
            (default) keeps observability off entirely — no subscription, no
            overhead. When provided it is attached to the event bus at
            construction time so it also captures ``agent.run_started``.
        hooks: Optional :class:`HookRegistry` with callbacks fired at each
            :class:`HookPoint` of the loop. ``None`` (default) means no hooks
            run at all (zero overhead). If the registry has no event bus of
            its own, the agent's bus is wired in so ``hooks.error`` events
            share the agent's observability stream. Hooks are also the
            extension point for a real confirmation UX: when the permission
            policy flags a call ``requires_confirmation=True``, the agent
            asks the ``ON_PERMISSION_CHECK`` hooks instead of denying
            outright — see :meth:`_execute_one`.
        stuck_detector: Optional sliding-window guard for repeated tool calls.
            When it detects repetition, the run is cooperatively cancelled
            after emitting ``agent.stuck_detected``. ``None`` preserves the
            historical behavior with no detection.

    Attributes:
        mode: Current :class:`AgentMode`. Set once by the classifier at the
            start of :meth:`run` (MAX when no classifier / on fallback). Only
            FLASH -> MAX escalation is allowed within a single task via
            :meth:`escalate`; the reverse is not.
        enable_extended_reasoning: Mode-driven flag. ``False`` in FLASH, ``True``
            in MAX. A placeholder switch for the future planner/verifier
            pipeline; the loop itself does not branch on it yet.
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
        llm: LLMClient,
        tools: Iterable[Tool] | None = None,
        state: ConversationState | None = None,
        event_bus: EventBus | None = None,
        classifier: TaskClassifier | None = None,
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
        stuck_detector: StuckDetector | None = None,
    ) -> None:
        self.llm = llm
        #: Optional persistence backend. When set (and no explicit ``state``
        #: was passed) the saved conversation is resumed at construction, and
        #: the state is persisted after every turn — a crash mid-run loses at
        #: most the in-flight turn. Store failures are logged, never fatal.
        self.state_store = state_store
        # NOTE: use ``is not None`` rather than truthiness because
        # ConversationState defines __len__ (an empty state is falsy but is
        # still a perfectly valid state object the caller passed in).
        if state is not None:
            self.state = state
        elif state_store is not None:
            resumed = state_store.load()
            self.state = resumed if resumed is not None else ConversationState()
        else:
            self.state = ConversationState()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.classifier: TaskClassifier = classifier if classifier is not None else DefaultClassifier()
        self.context_manager: ContextManager = (
            context_manager if context_manager is not None else SimpleTruncateContextManager()
        )
        if isinstance(self.context_manager, SummarizingContextManager) and (
            self.context_manager.event_bus is None
        ):
            # Route summarization-failure events through the agent's bus so
            # they land in the same observability stream as context.compacted.
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
        #: Per-call wall-clock limit for ``tool.execute``. ``None`` disables
        #: the guard (historical behaviour). On expiry the loop receives an
        #: error ToolResult and continues; Python cannot safely kill a running
        #: thread, so the timed-out call keeps running in the background -
        #: the timeout unblocks the agent, it does not cancel the work.
        self.tool_timeout = tool_timeout
        #: When True, a batch of tool calls in one model turn is fanned out to
        #: a thread pool instead of running strictly in order. Gating (hooks,
        #: policy, confirmation, validation) still runs sequentially on the
        #: main thread BEFORE anything executes; results are finalized
        #: (audit, AFTER hooks, state, events) in the model's original order.
        #: OPT-IN because tools must be thread-safe for it to be sound:
        #: built-in tools are (each call is an independent subprocess or file
        #: op), but a custom tool sharing mutable state is not.
        self.parallel_tool_execution = parallel_tool_execution
        self._executor: ThreadPoolExecutor | None = None
        #: Validate model-supplied arguments against each tool's declared
        #: JSON schema before executing (see ``tool/validation.py``). Invalid
        #: input becomes an error ToolResult with an actionable message
        #: instead of a confusing TypeError inside the tool.
        self.validate_tool_inputs = validate_tool_inputs
        #: Cooperative cancellation flag, set via :meth:`cancel` from any
        #: thread. Checked between iterations and between tool calls.
        self._cancel_event = threading.Event()
        #: Cooperative pause gate. Set means the loop may continue; clearing it
        #: pauses only at safe boundaries between completed tool-call rounds.
        self._pause_event = threading.Event()
        self._pause_event.set()
        #: Cross-thread user messages to add at the next replay-valid boundary.
        self._inbox: queue.Queue[str] = queue.Queue()
        #: UUID of the in-flight (or most recent) :meth:`run`; ``None`` before
        #: the first run. Every event emitted during a run carries it.
        self._run_id: str | None = None
        # User override of the iteration cap. ``None`` => let routing pick per
        # mode. Stored separately so an escalation can re-derive the MAX cap.
        self._max_iterations_override: int | None = max_iterations
        self.max_iterations: int = max_iterations if max_iterations is not None else self.MODE_MAX_ITERATIONS[AgentMode.MAX]
        self.mode: AgentMode = AgentMode.MAX
        self.enable_extended_reasoning: bool = True
        #: Once any run of this conversation classified MAX (or escalated to
        #: it), later runs never route back to FLASH. Set on successful MAX
        #: classification and on escalation - NOT on the exception fallback,
        #: so a transient classifier outage does not pin MAX forever.
        self._sticky_max: bool = False
        #: Running LLM usage totals across all runs of this agent (per-run
        #: data is in the ``llm.usage`` events; see ``llm/usage.py``).
        self.usage = UsageAccumulator()
        #: Parsed final answer of the last ``run(output_schema=...)`` call
        #: (None when no schema was requested or validation ultimately failed).
        self.structured_output: Any = None
        #: Optional long-term memory. When set, a run recalls relevant
        #: entries BEFORE the user turn (emits ``agent.memory_recalled``) and
        #: stores the Q/A pair AFTER it (emits ``agent.memory_stored``).
        self.memory = memory
        self.memory_recall_limit = memory_recall_limit
        #: Optional root-run cost guardrail (see ``agent/budget.py``). Checked
        #: before every LLM call; on exhaustion the run stops gracefully with
        #: an explanatory final text and an ``agent.budget_exceeded`` event —
        #: no exception escapes the loop. ``None`` (default) = unlimited.
        self.run_budget = run_budget
        #: Optional sliding-window repeated-tool-call guard. Disabled by default.
        self.stuck_detector = stuck_detector

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

    def run(
        self,
        user_message: str | None = None,
        *,
        stream: bool = False,
        output_schema: dict[str, Any] | None = None,
        structured_retries: int | None = None,
    ) -> str:
        """Run the agent loop until the model stops calling tools.

        Before the first turn the task is classified exactly once (see
        :attr:`classifier`); the result sets :attr:`mode` (FLASH for SIMPLE,
        MAX for COMPLEX, fallback MAX on any classifier failure) and the
        iteration cap for this run. The classifier is never re-invoked later in
        the same run, even after an escalation.

        Mid-run escalation FLASH -> MAX is triggered when:
        * the first turn's tool call(s) report an error, or
        * :attr:`FLASH_ESCALATION_THRESHOLD` iterations pass in FLASH without a
          final answer.
        Escalation keeps the conversation state, raises the cap to the MAX
        default, and emits ``agent.escalated`` exactly once. MAX -> FLASH is
        never performed.

        Args:
            user_message: Optional user turn to append before running. Pass
                ``None`` to continue an existing conversation (e.g. after a
                tool result injected externally - not used in Stage 1).
            stream: When ``True``, LLM turns use ``chat_stream()`` and every
                text chunk is published in real time as an ``agent.text_delta``
                event (payload: ``{"delta": str}``) instead of observers
                waiting for the whole response. Tool calling is unaffected —
                the model may stream text first and still end the turn with
                tool calls. Clients without streaming support fall back to a
                plain ``chat()`` call (logged, no deltas emitted). The return
                value is the same either way.
            output_schema: Optional JSON Schema (the same subset as
                ``tool/validation.py``) the FINAL answer must match. The
                schema is appended to the task as a natural-language
                instruction; answers that fail to parse/validate are sent
                back to the model as correction turns (up to
                ``structured_retries``). On success the parsed value is on
                :attr:`structured_output` (the return value stays the raw
                final text). Emitting events: ``agent.structured_output_parsed``,
                ``agent.structured_output_retry`` (per failed attempt),
                ``agent.structured_output_invalid`` (retries exhausted).
            structured_retries: Correction rounds for ``output_schema``
                (default :data:`~kinetic_sdk.agent.structured.DEFAULT_STRUCTURED_RETRIES`
                = 2).

        Returns:
            The final assistant text. If the loop hit ``max_iterations``
            without a final answer, returns the last assistant text seen
            (possibly empty) and publishes an ``agent.error`` event. If the
            run was cancelled via :meth:`cancel`, returns the best text seen
            so far (possibly empty) and publishes ``agent.cancelled``. If a
            :attr:`run_budget` was set and is exhausted, the loop stops before
            the next LLM call, emits ``agent.budget_exceeded`` and returns a
            final text starting with ``"Run stopped: budget exceeded"``.
        """
        self.structured_output = None
        if user_message is not None and self.memory is not None:
            self._recall_memory(user_message)
        composed = user_message
        if output_schema is not None:
            composed = (user_message or "") + "\n\n" + schema_instruction(output_schema)
        if composed is not None:
            self.state.add_user_message(composed)

        self._cancel_event.clear()  # a new run starts un-cancelled
        self._run_id = str(uuid.uuid4())
        self._trigger_hooks(
            HookPoint.BEFORE_RUN,
            HookContext(
                point=HookPoint.BEFORE_RUN, run_id=self._run_id, user_message=user_message
            ),
        )
        self._classify_and_route(user_message)

        self._emit(
            "agent.run_started",
            {"mode": self.mode.value, "tools": list(self._tools), "max_iterations": self.max_iterations},
        )

        final_text = ""
        try:
            final_text = self._run_loop(
                stream=stream,
                output_schema=output_schema,
                structured_retries=structured_retries,
            )
        except Exception as exc:
            self._trigger_hooks(
                HookPoint.ON_ERROR,
                HookContext(
                    point=HookPoint.ON_ERROR,
                    run_id=self._run_id,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            self._emit("agent.error", {"reason": "exception", "error": str(exc)})
            raise

        self._trigger_hooks(
            HookPoint.AFTER_RUN,
            HookContext(
                point=HookPoint.AFTER_RUN, run_id=self._run_id, final_text=final_text
            ),
        )
        self._emit("agent.run_finished", {"final_text": final_text, "mode": self.mode.value})
        if user_message is not None and self.memory is not None:
            self._store_memory(user_message, final_text)
        return final_text

    def escalate(self) -> bool:
        """Escalate from FLASH to MAX mid-task. Returns True if it escalated.

        On a successful escalation the iteration cap is raised to the MAX
        default (unless the caller pinned ``max_iterations`` explicitly), and
        :attr:`enable_extended_reasoning` is turned on. Downgrading is
        intentionally not supported within one task.
        """
        target = AgentMode.escalates_to(self.mode)
        if target is None:
            return False
        previous = self.mode
        self.mode = target
        self.enable_extended_reasoning = target is AgentMode.MAX
        if target is AgentMode.MAX:
            # Once a conversation has run at MAX it never drops back to
            # FLASH - a mid-conversation downgrade would silently shrink the
            # iteration budget of later turns.
            self._sticky_max = True
        if self._max_iterations_override is None:
            self.max_iterations = self.MODE_MAX_ITERATIONS[target]
        self._emit("agent.escalated", {"from": previous.value, "to": target.value})
        return True

    # --- internals ----------------------------------------------------

    def _classify_and_route(self, user_message: str | None) -> None:
        """Classify the task once and apply the mode-specific config."""
        task = user_message if user_message is not None else self._first_user_message() or ""
        try:
            classification = self.classifier.classify(task)
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
        self._emit(
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

    def cancel(self) -> None:
        """Ask the in-flight :meth:`run` to stop cooperatively.

        Thread-safe. The loop notices between iterations and between tool
        calls, fills in error results for any skipped tool calls (so the
        history stays replay-valid), emits ``agent.cancelled`` and returns
        the best final text it has. A tool call already executing is NOT
        interrupted — cooperative cancellation cannot preempt running work.
        """
        self._cancel_event.set()
        self._pause_event.set()

    def pause(self) -> None:
        """Pause an in-flight run at its next safe loop boundary.

        A currently executing tool is deliberately allowed to finish, keeping
        the conversation free of dangling ``tool_use`` blocks.
        """
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume a run paused by :meth:`pause` without resetting its state."""
        self._pause_event.set()

    def send_message_while_running(self, text: str) -> None:
        """Queue a user message for the next tool-call-round boundary.

        The method is non-blocking and thread-safe. Messages are appended by
        the run thread, immediately before its next LLM request.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        self._inbox.put(text)

    @property
    def cancelled(self) -> bool:
        """True after :meth:`cancel` until the next :meth:`run` starts."""
        return self._cancel_event.is_set()

    def _run_loop(
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
            self._pause_event.wait()
            if self._cancel_event.is_set():
                logger.info("Agent run cancelled at iteration %d", iteration)
                self._emit("agent.cancelled", {"iteration": iteration})
                return final_text
            self._emit("agent.turn_started", {"iteration": iteration, "mode": self.mode.value})
            self._maybe_compact_context()
            hook_results = self._trigger_hooks(
                HookPoint.BEFORE_LLM_CALL,
                HookContext(
                    point=HookPoint.BEFORE_LLM_CALL,
                    run_id=self._run_id,
                    iteration=iteration,
                    user_message=self._first_user_message(),
                ),
            )
            # Re-check after hooks: a hook is a common place for UI-driven
            # cancellation, and it should prevent the imminent LLM call.
            if self._cancel_event.is_set():
                logger.info("Agent run cancelled at iteration %d", iteration)
                self._emit("agent.cancelled", {"iteration": iteration})
                return final_text
            budget_stop = self._budget_stop_message()
            if budget_stop is not None:
                logger.warning("Run budget exceeded: %s", budget_stop)
                return budget_stop
            system_append = self._system_prompt_append(hook_results)
            response = self._call_llm(stream=stream, system_append=system_append)
            self._trigger_hooks(
                HookPoint.AFTER_LLM_CALL,
                HookContext(
                    point=HookPoint.AFTER_LLM_CALL,
                    run_id=self._run_id,
                    iteration=iteration,
                    llm_response=response,
                ),
            )
            self.state.add_assistant(self._assistant_content(response))
            self._emit(
                "agent.llm_response",
                {"tool_calls": len(response.tool_calls), "stop_reason": response.stop_reason},
            )

            if not response.tool_calls:
                final_text = response.content
                if output_schema is not None:
                    decision = self._check_structured(output_schema, final_text, retries_left)
                    if decision == "retry":
                        retries_left -= 1
                        iteration += 1
                        continue
                # Persist only at replay-valid boundaries: an assistant text
                # turn (no dangling tool_use) or right after tool results.
                self._persist_state()
                break

            any_error = self._execute_tool_calls(response.tool_calls)
            self._persist_state()

            # A tool-call round is complete, so this is both replay-valid and
            # the earliest point at which injected user input can be acted on.
            self._drain_inbox()
            self._pause_event.wait()
            if self._cancel_event.is_set():
                logger.info("Agent run cancelled after tool calls at iteration %d", iteration)
                self._emit("agent.cancelled", {"iteration": iteration})
                return final_text

            if not escalated and self.mode is AgentMode.FLASH:
                should_escalate = False
                if iteration == 0 and any_error:
                    should_escalate = True
                elif (iteration + 1) >= self.FLASH_ESCALATION_THRESHOLD:
                    should_escalate = True
                if should_escalate:
                    escalated = self.escalate()
            iteration += 1
        else:
            logger.warning("Agent hit max_iterations=%d", self.max_iterations)
            self._emit(
                "agent.error",
                {"reason": "max_iterations", "iterations": self.max_iterations},
            )
        return final_text

    def _drain_inbox(self) -> None:
        """Append queued cross-thread user messages and notify observers."""
        while True:
            try:
                message = self._inbox.get_nowait()
            except queue.Empty:
                return
            self.state.add_user_message(message)
            self._emit("agent.message_injected", {"message": message})

    def _first_user_message(self) -> str | None:
        """Return the most recent user text message, if any (for re-runs)."""
        for msg in reversed(self.state.messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None

    def _maybe_compact_context(self) -> None:
        """Compact the history before an LLM call when over the threshold.

        Emits ``context.compacted`` with the before/after message counts so
        observability can trace when and how much was elided. The manager
        returns a new state (immutable-style); the agent swaps its reference.
        """
        manager = self.context_manager
        if manager is None or not manager.should_compact(self.state, self.model_context_limit):
            return
        before = len(self.state.messages)
        self.state = manager.compact(self.state)
        after = len(self.state.messages)
        logger.info("Context compacted: %d -> %d messages", before, after)
        self._emit(
            "context.compacted",
            {
                "manager": type(manager).__name__,
                "messages_before": before,
                "messages_after": after,
                "messages_removed": before - after,
                "estimated_tokens_after": manager.estimate_state_tokens(self.state),
            },
        )

    def _call_llm(
        self, stream: bool = False, system_append: str | None = None
    ) -> LLMResponse:
        """Ask the LLM for the next turn using the current history + tools.

        With ``stream=True`` the client's ``chat_stream`` is consumed instead:
        each text chunk is re-published as an ``agent.text_delta`` event as it
        arrives, and the aggregated response from the final ``done`` event is
        returned, so the rest of the loop (tool calling, escalation) works
        unchanged. A client that does not implement streaming falls back to a
        plain ``chat`` call.
        """
        system, messages = self.state.for_llm()
        if system_append:
            system = f"{system}\n\n{system_append}" if system else system_append
        tools = self.tool_schemas() or None
        if not stream:
            response = self.llm.chat(messages=messages, tools=tools, system=system)
            self._record_usage(response)
            return response
        try:
            final: LLMResponse | None = None
            for event in self.llm.chat_stream(messages=messages, tools=tools, system=system):
                if event.type == "text" and isinstance(event.delta, str):
                    self._emit("agent.text_delta", {"delta": event.delta})
                elif event.type == "done" and isinstance(event.delta, LLMResponse):
                    final = event.delta
        except NotImplementedError:
            logger.warning(
                "%s does not support streaming; falling back to chat()",
                type(self.llm).__name__,
            )
            response = self.llm.chat(messages=messages, tools=tools, system=system)
            self._record_usage(response)
            return response
        if final is None:
            # Stream ended without a done event: treat as an empty end turn
            # rather than crashing the loop.
            final = LLMResponse(content="", stop_reason="end_turn")
        self._record_usage(final)
        return final

    def _check_structured(
        self, schema: dict[str, Any], text: str, retries_left: int
    ) -> str:
        """Validate a final answer against *schema*; return "break" | "retry".

        Success sets :attr:`structured_output` and emits
        ``agent.structured_output_parsed``; failure with rounds left appends
        a correction user message and emits ``agent.structured_output_retry``;
        failure with no rounds left emits ``agent.structured_output_invalid``
        (the raw text is still returned as the final answer).
        """
        ok, value, problems = check_final_answer(schema, text)
        if ok:
            self.structured_output = value
            self._emit("agent.structured_output_parsed", {"schema": schema})
            return "break"
        if retries_left > 0:
            self.state.add_user_message(
                CORRECTION_TEMPLATE.format(problems="; ".join(problems))
            )
            self._emit("agent.structured_output_retry", {"problems": problems})
            return "retry"
        self._emit("agent.structured_output_invalid", {"problems": problems})
        return "break"

    def _recall_memory(self, query: str) -> None:
        """Inject relevant memories as a user message before the real task.

        The recall message lands in the history BEFORE the user's own
        message, so the model sees context first. Fail-soft (a broken
        provider never blocks the run): the failure is logged and the event
        ``agent.memory_recall_failed`` is emitted.
        """
        assert self.memory is not None
        try:
            recalled = self.memory.search(query, limit=self.memory_recall_limit)
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("Memory recall failed: %s", exc)
            self._emit(
                "agent.memory_recall_failed",
                {"error": redact_secrets(f"{type(exc).__name__}: {exc}")},
            )
            return
        if not recalled:
            return
        lines = "\n".join("- " + e.text for e in recalled)
        self.state.add_user_message(f"[Recalled memories]\n{lines}")
        self._emit("agent.memory_recalled", {"query": query, "count": len(recalled)})

    def _store_memory(self, user_message: str, final_text: str) -> None:
        """Remember the completed Q/A pair; fail-soft like recall."""
        assert self.memory is not None
        try:
            entry = self.memory.add(
                f"User: {user_message}\nAssistant: {final_text}",
                metadata={"run_id": self._run_id},
            )
            self._emit("agent.memory_stored", {"id": entry.id})
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("Memory store failed: %s", exc)
            self._emit(
                "agent.memory_store_failed",
                {"error": redact_secrets(f"{type(exc).__name__}: {exc}")},
            )

    def _budget_stop_message(self) -> str | None:
        """Check the run budget before an LLM call.

        Returns the graceful-stop final text when the budget is exhausted
        (emitting ``agent.budget_exceeded`` with used/limit details), or
        ``None`` when the next call may proceed. The exception raised by
        :meth:`RunBudget.check` never escapes the loop.
        """
        if self.run_budget is None:
            return None
        try:
            self.run_budget.check()
            return None
        except RunBudgetExceeded as exc:
            self._emit("agent.budget_exceeded", self.run_budget.details())
            return f"Run stopped: budget exceeded ({exc})."

    def _record_usage(self, response: LLMResponse) -> None:
        """Fold one LLM turn's usage into the agent total + emit a delta."""
        if self.run_budget is not None:
            # Count the call even when the provider reports no usage, so a
            # call-limited budget cannot be bypassed by a usage-silent client.
            self.run_budget.record(response.usage)
        if not response.usage:
            return
        self.usage.record(response.usage)
        # Delta only — totals live in ``agent.usage`` (cumulative across runs).
        self._emit("llm.usage", {"usage": dict(response.usage)})

    def _persist_state(self) -> None:
        """Save the conversation to the configured store, never fatally.

        Only called at replay-valid boundaries (assistant text turn or right
        after tool results), so a resumed history never ends on a dangling
        ``tool_use`` — providers reject those outright. A broken store (disk
        full, permissions) must not kill a run - the failure is logged and
        the loop continues with the in-memory state.
        """
        if self.state_store is None:
            return
        try:
            self.state_store.save(self.state)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to persist conversation state: %s", exc)
            self._emit(
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

    def _execute_tool_calls(self, calls: list[ToolCall]) -> bool:
        """Execute each tool call in order and append results to the state.

        Returns ``True`` if at least one tool call resulted in an error (used
        by the escalation logic to detect a failing first turn).

        When the run is cancelled mid-batch the remaining calls are NOT
        executed, but each still gets an error ``tool_result`` appended —
        every ``tool_use`` in the history must have its result or providers
        reject the whole conversation on the next call.
        """
        if self.parallel_tool_execution and len(calls) > 1:
            return self._execute_tool_calls_parallel(calls)
        any_error = False
        for call in calls:
            if self._cancel_event.is_set():
                result = ToolResult(error="skipped: run cancelled")
                self.state.add_tool_result(
                    call.id, self._format_tool_output(result), is_error=True
                )
                any_error = True
                continue
            self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            result = self._execute_one(call)
            any_error = any_error or result.is_error
            self._emit_tool_call_finished(call, result)
            self.state.add_tool_result(
                call.id,
                self._format_tool_output(result),
                is_error=result.is_error,
            )
        return any_error

    def _execute_tool_calls_parallel(self, calls: list[ToolCall]) -> bool:
        """Fan a multi-call batch out to the thread pool.

        Ordering guarantees, identical to the sequential path from the
        outside: gating happens per call in order (main thread), results are
        finalized and appended to the state in the model's original order,
        and every event is emitted from the main thread. Only
        ``tool.execute`` itself runs concurrently — a cancelled run still
        cannot preempt in-flight executions, and skipped calls still receive
        their error ``tool_result`` so the history stays valid.
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
            self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            tool, arguments, early = self._prepare_call(call)
            if early is not None:
                early_or_skip[i] = early
            else:
                assert tool is not None
                pending.append((i, call, tool, arguments))

        executed: dict[int, ToolResult] = {}
        executed_args: dict[int, dict[str, Any]] = {}
        if pending:
            executor = self._get_executor()
            futures = {
                executor.submit(tool.execute, **arguments): (i, call, arguments)
                for (i, call, tool, arguments) in pending
            }
            for future, (i, call, arguments) in futures.items():
                executed_args[i] = arguments
                try:
                    executed[i] = (
                        future.result(timeout=self.tool_timeout)
                        if self.tool_timeout is not None
                        else future.result()
                    )
                except FutureTimeoutError:
                    executed[i] = ToolResult(
                        error=(
                            f"Tool {call.name!r} timed out after {self.tool_timeout}s "
                            "(the call may still be running in the background)"
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - surface as tool error
                    logger.exception("Tool %s raised", call.name)
                    executed[i] = ToolResult(error=f"{type(exc).__name__}: {exc}")

        for i, call in enumerate(calls):
            if i in early_or_skip:
                result = early_or_skip[i]
                any_error = any_error or result.is_error
                if i not in skipped:
                    self._emit_tool_call_finished(call, result)
                self.state.add_tool_result(
                    call.id, self._format_tool_output(result), is_error=result.is_error
                )
                continue
            result = executed[i]
            any_error = any_error or result.is_error
            self._finalize_call(call, executed_args[i], result)
            self._emit_tool_call_finished(call, result)
            self.state.add_tool_result(
                call.id, self._format_tool_output(result), is_error=result.is_error
            )
        return any_error

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create the shared tool-execution thread pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="kinetic-tool"
            )
        return self._executor

    def _emit_tool_call_finished(self, call: ToolCall, result: ToolResult) -> None:
        self._emit(
            "agent.tool_call_finished",
            {
                "name": call.name,
                "id": call.id,
                "is_error": result.is_error,
                "output_preview": redact_secrets(self._preview(result.output)),
            },
        )
        if self.stuck_detector is not None and self.stuck_detector.observe(
            call.name, call.arguments
        ):
            self._emit(
                "agent.stuck_detected",
                {"tool_name": call.name, "window": self.stuck_detector.window},
            )
            self.cancel()

    @staticmethod
    def _system_prompt_append(results: list[HookResult]) -> str | None:
        """Collect request-local system-prompt additions from LLM hooks."""
        additions = [
            result.modified_context["system_prompt"]
            for result in results
            if isinstance(result, HookResult)
            and result.modified_context
            and isinstance(result.modified_context.get("system_prompt"), str)
            and result.modified_context["system_prompt"].strip()
        ]
        return "\n\n".join(additions) or None

    def _execute_one(self, call: ToolCall) -> ToolResult:
        """Dispatch a single tool call, gated by hooks + permission policy.

        Flow per call:

        1. ``BEFORE_TOOL_CALL`` hooks run. A hook may cancel the call
           (``should_continue=False``) or replace the input via
           ``modified_context={"tool_input": {...}}`` — the replacement is
           what gets policy-checked, audited and executed.
        2. The permission policy checks the call; the decision is audit-logged.
        3. ``requires_confirmation=True`` decisions are the Confirmation UX
           extension point: the ``ON_PERMISSION_CHECK`` hooks are consulted,
           and any hook answering ``should_continue=True`` counts as an
           explicit confirmation (e.g. a CLI prompt the user answered "yes").
           With no hooks configured, no hook registered at that point, or all
           hooks declining, the historical safe fallback applies — the call is
           denied with an explanatory message. The SDK core deliberately does
           not ship a concrete confirmation UI; it only provides the hook
           point (see ``kinetic_sdk.security`` for an ``input()``-based
           example).
        4. Allowed calls execute; the result is audit-logged and the
           ``AFTER_TOOL_CALL`` hooks run. Denied/cancelled calls return an
           error :class:`ToolResult` so the model can react, and denials
           emit ``security.permission_denied``.
        """
        tool, arguments, early = self._prepare_call(call)
        if early is not None:
            return early
        assert tool is not None  # guaranteed by _prepare_call's contract
        try:
            result = self._run_tool(tool, arguments)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("Tool %s raised", call.name)
            result = ToolResult(error=f"{type(exc).__name__}: {exc}")
        self._finalize_call(call, arguments, result)
        return result

    def _prepare_call(
        self, call: ToolCall
    ) -> tuple[Tool | None, dict[str, Any], ToolResult | None]:
        """Run pre-execution gating for one call, on the CALLER's thread.

        Steps 1-4 of :meth:`_execute_one`: BEFORE_TOOL_CALL hooks, permission
        policy + audit, confirmation fallback, unknown-tool check, and input
        schema validation. Returns ``(tool, arguments, None)`` when the call
        may execute, or ``(None, arguments, error_result)`` when it was
        short-circuited. Pure of thread-unsafe work: safe to run for a whole
        batch before executions are fanned out to the pool.
        """
        arguments = call.arguments
        if self.hooks is not None:
            results = self._trigger_hooks(
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
            return None, arguments, self._deny_call(call, decision.reason, arguments)
        if decision.requires_confirmation and not self._confirmed_by_hooks(
            call, arguments, decision
        ):
            return None, arguments, self._deny_call(
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

    def _finalize_call(
        self, call: ToolCall, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """Audit + AFTER_TOOL_CALL hooks for an EXECUTED call (main thread)."""
        self.audit_logger.log_tool_result(call.name, result, _utcnow())
        self._trigger_hooks(
            HookPoint.AFTER_TOOL_CALL,
            HookContext(
                point=HookPoint.AFTER_TOOL_CALL,
                run_id=self._run_id,
                tool_name=call.name,
                tool_input=arguments,
                tool_result=result,
            ),
        )

    def _run_tool(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        """Run ``tool.execute`` directly, or under a wall-clock timeout.

        With ``tool_timeout`` set the call runs on a shared thread pool and
        the loop only waits up to the timeout. A timed-out call returns an
        error result so the model can react; the worker thread itself is NOT
        killed (Python has no safe thread kill), so a tool that hangs forever
        still leaks its thread - the timeout protects the loop, not the
        process. Tools wrapping subprocesses/network should enforce their own
        real timeout too (``GitTool`` already does).
        """
        if self.tool_timeout is None:
            return tool.execute(**arguments)
        future = self._get_executor().submit(tool.execute, **arguments)
        try:
            return future.result(timeout=self.tool_timeout)
        except FutureTimeoutError:
            return ToolResult(
                error=(
                    f"Tool {tool.name!r} timed out after {self.tool_timeout}s "
                    "(the call may still be running in the background)"
                )
            )

    def _trigger_hooks(self, point: HookPoint, context: HookContext) -> list[HookResult]:
        """Run the hooks registered for *point*, or nothing when unconfigured."""
        if self.hooks is None:
            return []
        return self.hooks.trigger(point, context)

    def _confirmed_by_hooks(
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
        results = self._trigger_hooks(
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

    def _deny_call(
        self, call: ToolCall, reason: str, tool_input: dict[str, Any] | None = None
    ) -> ToolResult:
        """Audit-log + emit a denial and build the error result for the model."""
        tool_input = tool_input if tool_input is not None else call.arguments
        logger.warning("Permission denied for tool %s: %s", call.name, reason)
        self.audit_logger.log_permission_denied(call.name, tool_input, reason, _utcnow())
        self._emit(
            "security.permission_denied",
            {
                "name": call.name,
                "id": call.id,
                "reason": reason,
                "input_preview": redact_secrets(self._preview(tool_input)),
            },
        )
        return ToolResult(error=f"Permission denied for tool {call.name!r}: {reason}")

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

    @property
    def run_id(self) -> str | None:
        """UUID of the current/last :meth:`run`, or ``None`` before any run."""
        return self._run_id

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a lifecycle event if a bus is attached.

        When a run is in flight its ``run_id`` is added to the payload so
        observability subscribers can group events per run. Existing payload
        fields are left untouched.
        """
        if self.event_bus is None:
            return
        if self._run_id is not None:
            payload = {**payload, "run_id": self._run_id}
        self.event_bus.publish(Event(type=event_type, payload=payload, source="agent"))
