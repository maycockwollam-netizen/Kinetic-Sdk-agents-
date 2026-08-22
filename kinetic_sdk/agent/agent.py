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
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from kinetic_sdk.agent.classifier import TaskClassifier, DefaultClassifier
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.context.manager import (
    ContextManager,
    SimpleTruncateContextManager,
    SummarizingContextManager,
)
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import EventBus, Event
from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.hooks.registry import HookRegistry
from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
from kinetic_sdk.observability.logger import ObservabilityLogger
from kinetic_sdk.security.audit import AuditLogger, InMemoryAuditLogger
from kinetic_sdk.security.policy import AllowListPolicy, PermissionDecision, PermissionPolicy
from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.tool.base import Tool, ToolResult

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
              ``agent.llm_response``, ``agent.tool_call_started``,
              ``agent.tool_call_finished``, ``agent.run_finished``,
              ``agent.escalated``, ``agent.classified``, ``agent.error``,
              ``context.compacted``, ``context.summarization_failed``,
              ``security.permission_denied``, ``hooks.error`` (via the
              hook registry).
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
    ) -> None:
        self.llm = llm
        # NOTE: use ``is not None`` rather than truthiness because
        # ConversationState defines __len__ (an empty state is falsy but is
        # still a perfectly valid state object the caller passed in).
        self.state = state if state is not None else ConversationState()
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
        #: UUID of the in-flight (or most recent) :meth:`run`; ``None`` before
        #: the first run. Every event emitted during a run carries it.
        self._run_id: str | None = None
        # User override of the iteration cap. ``None`` => let routing pick per
        # mode. Stored separately so an escalation can re-derive the MAX cap.
        self._max_iterations_override: int | None = max_iterations
        self.max_iterations: int = max_iterations if max_iterations is not None else self.MODE_MAX_ITERATIONS[AgentMode.MAX]
        self.mode: AgentMode = AgentMode.MAX
        self.enable_extended_reasoning: bool = True

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

    def run(self, user_message: str | None = None) -> str:
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

        Returns:
            The final assistant text. If the loop hit ``max_iterations``
            without a final answer, returns the last assistant text seen
            (possibly empty) and publishes an ``agent.error`` event.
        """
        if user_message is not None:
            self.state.add_user_message(user_message)

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
            final_text = self._run_loop()
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
        self._apply_mode(classification.mode, rationale=classification.rationale)
        self._emit(
            "agent.classified",
            {
                "complexity": classification.complexity.value,
                "mode": classification.mode.value,
                "confidence": classification.confidence,
                "rationale": classification.rationale,
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

    def _run_loop(self) -> str:
        """The tool-calling loop, with mid-run FLASH -> MAX escalation."""
        final_text = ""
        iteration = 0
        escalated = False
        while iteration < self.max_iterations:
            self._emit("agent.turn_started", {"iteration": iteration, "mode": self.mode.value})
            self._maybe_compact_context()
            self._trigger_hooks(
                HookPoint.BEFORE_LLM_CALL,
                HookContext(
                    point=HookPoint.BEFORE_LLM_CALL,
                    run_id=self._run_id,
                    iteration=iteration,
                ),
            )
            response = self._call_llm()
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
                break

            any_error = self._execute_tool_calls(response.tool_calls)

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

    def _call_llm(self) -> LLMResponse:
        """Ask the LLM for the next turn using the current history + tools."""
        system, messages = self.state.for_llm()
        tools = self.tool_schemas() or None
        return self.llm.chat(messages=messages, tools=tools, system=system)

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
        """
        any_error = False
        for call in calls:
            self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            result = self._execute_one(call)
            any_error = any_error or result.is_error
            self._emit(
                "agent.tool_call_finished",
                {
                    "name": call.name,
                    "id": call.id,
                    "is_error": result.is_error,
                    "output_preview": redact_secrets(self._preview(result.output)),
                },
            )
            self.state.add_tool_result(
                call.id,
                self._format_tool_output(result),
                is_error=result.is_error,
            )
        return any_error

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
                return ToolResult(
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
            return self._deny_call(call, decision.reason, arguments)
        if decision.requires_confirmation and not self._confirmed_by_hooks(
            call, arguments, decision
        ):
            return self._deny_call(
                call,
                "requires manual confirmation, not yet supported in automated "
                f"mode ({decision.reason})",
                arguments,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            logger.error("Unknown tool requested: %s", call.name)
            return ToolResult(error=f"Unknown tool: {call.name}")
        try:
            result = tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("Tool %s raised", call.name)
            result = ToolResult(error=f"{type(exc).__name__}: {exc}")
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
        return result

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
