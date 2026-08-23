"""Spawning sub-agents: the delegation core.

``spawn_subagent`` builds a NEW :class:`~kinetic_sdk.agent.agent.Agent` from
a parent agent and a :class:`SubagentSpec`, following three locked design
decisions (do not "fix" these — they are deliberate):

1. **Full inheritance, no narrowing.** The sub-agent receives the parent's
   ENTIRE tool set and the SAME ``permission_policy`` object (the pattern
   "narrow controller -> executor that needs wider tools to do real work"
   must stay possible). The single deliberate difference from an exact
   copy of the parent: the sub-agent gets its OWN ``system_prompt`` from
   the spec, never the parent's.
2. **No depth cap.** Delegation is an ordinary tool, so sub-agents may
   spawn sub-sub-agents without a hard nesting limit. Runaway recursion is
   instead bounded by cost/behaviour guardrails (decision 3).
3. **Guardrails by budget + circuit breaker, not by depth.** One shared
   :class:`SpawnBudget` is threaded through the whole tree; each sub-agent
   additionally gets its OWN :class:`RepetitionCircuitBreaker`.

**How the guardrails are enforced** (without touching ``Agent.run``): the
sub-agent's LLM client is wrapped in a guard that charges every requested
tool call to the shared budget and to the per-agent breaker BEFORE the
calls execute. An exhausted budget / tripped breaker raises out of the LLM
call — a path the agent loop deliberately does not swallow — so the
sub-agent's ``run()`` fails fast instead of looping until
``max_iterations``. ``DelegateTool`` one level up converts the failure
into an error ``ToolResult``.

**Context isolation.** The sub-agent starts from a brand-new
:class:`ConversationState` (only ``spec.system_prompt``); the parent ->
child channel is exactly ONE ``task_prompt`` string, and the child ->
parent channel is exactly the final message inside :class:`DelegationResult`.
Intermediate tool calls of the sub-agent never leak into the parent's
context.
"""

from __future__ import annotations

import logging
import threading
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event
from kinetic_sdk.llm.client import LLMClient, LLMResponse, Message, StreamEvent
from kinetic_sdk.subagent.budget import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    RepetitionCircuitBreaker,
    SpawnBudget,
)
from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
    SubagentError,
)
from kinetic_sdk.subagent.manifest import SubagentSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.subagent.tool import DelegateTool

logger = logging.getLogger(__name__)

#: Factory building an LLM client for a requested model name. Needed only
#: when ``SubagentSpec.model`` differs from the parent's model — the SDK
#: cannot know how to construct provider clients generically.
LLMFactory = Callable[[str], LLMClient]

#: Stable ids for agents participating in delegation (the audit trail uses
#: them to reconstruct the parent/child tree). Keyed by identity; entries
#: die with their agent.
_AGENT_IDS: "weakref.WeakKeyDictionary[Agent, str]" = weakref.WeakKeyDictionary()
_AGENT_IDS_LOCK = threading.Lock()


def agent_id_for(agent: Agent) -> str:
    """Return the stable delegation id of *agent*, minting one on first use.

    The id is what ``subagent_spawn`` audit entries record as
    ``parent_agent_id`` / ``agent_id`` — the only evidence needed to rebuild
    the delegation tree after the fact.
    """
    with _AGENT_IDS_LOCK:
        known = _AGENT_IDS.get(agent)
        if known is None:
            known = uuid.uuid4().hex
            _AGENT_IDS[agent] = known
        return known


@dataclass(frozen=True)
class DelegationResult:
    """What the parent receives after a sub-agent run: the outcome boundary.

    Attributes:
        final_message: ONLY the sub-agent's last assistant text — never the
            transcript, so intermediate tool calls cannot leak into the
            parent's context.
        agent_id: Stable id of the spawned sub-agent (matches the audit
            trail's ``agent_id``).
    """

    final_message: str
    agent_id: str


class _GuardedLLMClient(LLMClient):
    """LLM client wrapper charging tool calls to the budget + breaker.

    Counts calls the model REQUESTS (not just ones the policy later lets
    execute): the LLM turn has already cost money either way, and a denied
    call in a loop is still runaway behaviour worth stopping. The raises
    happen inside ``chat`` — a call path ``Agent.run`` re-raises rather
    than converts to a ``ToolResult`` — which is what makes the guardrails
    fail-fast.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        budget: SpawnBudget,
        breaker: RepetitionCircuitBreaker,
        agent_id: str,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._breaker = breaker
        self._agent_id = agent_id

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    @property
    def inner(self) -> LLMClient:
        """The wrapped client (escape hatch for provider-specific access)."""
        return self._inner

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self._inner.chat(messages, tools=tools, system=system, **kwargs)
        for call in response.tool_calls:
            self._budget.record_tool_call(self._agent_id, call.name)
            self._breaker.record(call.name, call.arguments)
        return response

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        # The agent loop only uses chat(); streaming delegates unguarded in
        # this version (tool calls would only be visible once aggregated).
        return self._inner.chat_stream(messages, tools=tools, system=system, **kwargs)


def _audit_spawn(
    parent_agent: Agent,
    spec: SubagentSpec,
    *,
    agent_id: str,
    model: str | None,
    budget: SpawnBudget,
) -> None:
    parent_agent.audit_logger.log_event(
        "subagent_spawn",
        spec.name,
        datetime.now(timezone.utc),
        parent_agent_id=agent_id_for(parent_agent),
        agent_id=agent_id,
        spec_name=spec.name,
        model=model,
        budget_used=budget.used,
        budget_max=budget.max_total_tool_calls,
    )


def _audit_finish(
    parent_agent: Agent,
    spec: SubagentSpec,
    agent_id: str,
    outcome: str,
    detail: str,
) -> None:
    parent_agent.audit_logger.log_event(
        "subagent_finished",
        spec.name,
        datetime.now(timezone.utc),
        agent_id=agent_id,
        spec_name=spec.name,
        outcome=outcome,
        detail=detail,
    )


def spawn_subagent(
    parent_agent: Agent,
    spec: SubagentSpec,
    budget: SpawnBudget,
    *,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
    llm_factory: LLMFactory | None = None,
) -> Agent:
    """Build the sub-agent for *spec* (does NOT run it — see ``run_subagent``).

    Args:
        parent_agent: The agent delegating work. Its tools, permission
            policy, audit logger, event bus, hooks, classifier and context
            configuration are inherited; its conversation history and system
            prompt are NOT.
        spec: What to spawn. ``spec.system_prompt`` becomes the sub-agent's
            system message; ``spec.model`` optionally reroutes the model.
        budget: The SHARED tree budget. It is handed to the sub-agent (and,
            through the cloned ``DelegateTool`` instances, to anything the
            sub-agent spawns in turn) — never copied.
        max_consecutive_repeats: Per-agent circuit-breaker limit for the new
            sub-agent (its OWN breaker, not shared with the parent).
        llm_factory: Required only when ``spec.model`` names a model other
            than the parent's; called with the model string to build the
            sub-agent's client. Without it, a mismatched ``spec.model``
            raises :class:`SubagentError` rather than silently running on
            the parent's model.

    Every spawn is audit-logged (``subagent_spawn``) on the parent's
    (shared) audit logger — including spawns whose run later gets blocked
    by the budget or breaker, which are paired with a ``subagent_finished``
    entry naming the outcome.
    """
    base_llm = parent_agent.llm
    if isinstance(base_llm, _GuardedLLMClient):
        # Never stack guards: when a sub-agent spawns its own sub-agent, the
        # parent's client is already guarded — wrapping it again would charge
        # every call twice and would feed the child's calls into the parent's
        # circuit breaker (breakers must stay strictly per-agent).
        base_llm = base_llm.inner
    if spec.model is not None and spec.model != getattr(base_llm, "model", None):
        if llm_factory is None:
            raise SubagentError(
                f"Sub-agent {spec.name!r} requests model {spec.model!r}, which "
                "differs from the parent's model; pass llm_factory to build a "
                "client for it (or leave spec.model=None to inherit)"
            )
        base_llm = llm_factory(spec.model)

    agent_id = uuid.uuid4().hex
    guarded_llm = _GuardedLLMClient(
        base_llm,
        budget=budget,
        breaker=RepetitionCircuitBreaker(max_consecutive_repeats),
        agent_id=agent_id,
    )

    # Imported lazily: tool.py imports this module (DelegateTool ->
    # spawn_subagent), so a top-level import would be circular.
    from kinetic_sdk.subagent.tool import DelegateTool

    # Full tool inheritance (design decision 1). DelegateTools are cloned
    # and re-bound to the SUB-agent (sharing the same spec registry and the
    # SAME budget instance) so nested spawns charge the shared budget too.
    tools: list[Any] = []
    delegate_clones: list[DelegateTool] = []
    for tool in parent_agent._tools.values():  # noqa: SLF001 - intra-SDK access
        if isinstance(tool, DelegateTool):
            clone = tool._clone_for(budget)  # noqa: SLF001
            delegate_clones.append(clone)
            tools.append(clone)
        else:
            tools.append(tool)

    sub_agent = Agent(
        llm=guarded_llm,
        tools=tools,
        # Fresh context: own system prompt, no parent history (decision 1 +
        # context isolation). The ONLY parent -> child channel is the
        # task_prompt passed to run() later.
        state=ConversationState(system_prompt=spec.system_prompt),
        event_bus=parent_agent.event_bus,
        classifier=parent_agent.classifier,
        max_iterations=parent_agent._max_iterations_override,  # noqa: SLF001
        context_manager=parent_agent.context_manager,
        model_context_limit=parent_agent.model_context_limit,
        permission_policy=parent_agent.permission_policy,
        audit_logger=parent_agent.audit_logger,
        # The bus is shared with the parent, whose observability logger (if
        # any) is already attached to it — attaching again would double-log.
        observability_logger=None,
        hooks=parent_agent.hooks,
    )
    for clone in delegate_clones:
        clone.bind(sub_agent)

    with _AGENT_IDS_LOCK:
        _AGENT_IDS[sub_agent] = agent_id
    _audit_spawn(
        parent_agent,
        spec,
        agent_id=agent_id,
        model=guarded_llm.model,
        budget=budget,
    )
    sub_agent.event_bus.publish(
        Event(
            type="subagent.spawned",
            payload={
                "spec_name": spec.name,
                "agent_id": agent_id,
                "parent_agent_id": agent_id_for(parent_agent),
            },
            source="subagent",
        )
    )
    return sub_agent


def run_subagent(
    parent_agent: Agent,
    spec: SubagentSpec,
    task_prompt: str,
    budget: SpawnBudget,
    *,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
    llm_factory: LLMFactory | None = None,
) -> DelegationResult:
    """Spawn the sub-agent and run it to completion on *task_prompt*.

    Returns a :class:`DelegationResult` carrying ONLY the sub-agent's final
    message. Guardrail failures (:class:`BudgetExceededError`,
    :class:`RepetitionLimitError`) propagate to the caller — turning them
    into a ``ToolResult`` is ``DelegateTool``'s job, so direct callers can
    also handle them programmatically. Every outcome (completed, blocked,
    crashed) is written to the shared audit log as ``subagent_finished``.
    """
    sub_agent = spawn_subagent(
        parent_agent,
        spec,
        budget,
        max_consecutive_repeats=max_consecutive_repeats,
        llm_factory=llm_factory,
    )
    agent_id = agent_id_for(sub_agent)
    try:
        final_message = sub_agent.run(task_prompt)
    except BudgetExceededError as exc:
        _audit_finish(parent_agent, spec, agent_id, "budget_exceeded", str(exc))
        raise
    except RepetitionLimitError as exc:
        _audit_finish(parent_agent, spec, agent_id, "repetition_limit", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - audit before re-raising
        _audit_finish(
            parent_agent, spec, agent_id, "error", f"{type(exc).__name__}: {exc}"
        )
        raise
    _audit_finish(parent_agent, spec, agent_id, "completed", final_message)
    return DelegationResult(final_message=final_message, agent_id=agent_id)
