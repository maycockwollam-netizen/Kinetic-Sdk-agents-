"""Async sub-agent spawning: the async twin of ``delegation.py``.

Same three locked design decisions as the sync path (full tool/policy
inheritance, no depth cap, budget + circuit breaker instead of depth),
same context-isolation rule (fresh conversation, one ``task_prompt`` in,
one final message out). Only the mechanics differ: the sub-agent is an
:class:`~kinetic_sdk.agent.async_agent.AsyncAgent`, its LLM client is
wrapped in :class:`_GuardedAsyncLLMClient`, and the run is awaited.

The guard charges every requested tool call to the SHARED tree budget and
the per-agent breaker inside ``chat()`` — the async loop re-raises out of
``_call_llm`` just like the sync one, so guardrails still fail fast, and
``AsyncDelegateTool`` one level up converts the failure into an error
``ToolResult``.
"""

from __future__ import annotations

import logging
import threading
import uuid
import weakref
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import Event
from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LLMClient,
    LLMResponse,
    Message,
    StreamEvent,
)
from kinetic_sdk.subagent.budget import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    RepetitionCircuitBreaker,
    SpawnBudget,
)
from kinetic_sdk.subagent.delegation import DelegationResult
from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
    SubagentError,
)
from kinetic_sdk.subagent.manifest import SubagentSpec

logger = logging.getLogger(__name__)

#: Factory building a client for ``SubagentSpec.model`` when it differs
#: from the parent's. The result may be either client flavour — AsyncAgent
#: wraps sync clients automatically.
AsyncLLMFactory = Callable[[str], AsyncLLMClient | LLMClient]

_AGENT_IDS: "weakref.WeakKeyDictionary[AsyncAgent, str]" = weakref.WeakKeyDictionary()
_AGENT_IDS_LOCK = threading.Lock()


def async_agent_id_for(agent: AsyncAgent) -> str:
    """Stable delegation id for an :class:`AsyncAgent` (audit tree building)."""
    with _AGENT_IDS_LOCK:
        known = _AGENT_IDS.get(agent)
        if known is None:
            known = uuid.uuid4().hex
            _AGENT_IDS[agent] = known
        return known


class _GuardedAsyncLLMClient(AsyncLLMClient):
    """Async LLM client wrapper charging tool calls to budget + breaker.

    Mirrors the sync guard: calls are charged when REQUESTED (a later
    policy denial does not refund the LLM turn), and the raise happens
    inside ``chat`` — a path the async loop re-raises — which is what
    makes the guardrails fail-fast instead of loop-until-max_iterations.
    """

    def __init__(
        self,
        inner: AsyncLLMClient,
        *,
        budget: SpawnBudget,
        breaker: RepetitionCircuitBreaker,
        agent_id: str,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._breaker = breaker
        self._agent_id = agent_id
        # Plain attribute (not a property) so the guard stays assignment-
        # compatible with the writeable ``model`` attribute on the ABC.
        self.model = getattr(inner, "model", "unknown")

    @property
    def inner(self) -> AsyncLLMClient:
        """The wrapped client (escape hatch + guard un-stacking)."""
        return self._inner

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = await self._inner.chat(messages, tools=tools, system=system, **kwargs)
        for call in response.tool_calls:
            self._budget.record_tool_call(self._agent_id, call.name)
            self._breaker.record(call.name, call.arguments)
        return response

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        # Unguarded like the sync guard: the loop uses chat(); streamed tool
        # calls would only be visible once aggregated anyway.
        async for event in self._inner.chat_stream(
            messages, tools=tools, system=system, **kwargs
        ):
            yield event


def spawn_async_subagent(
    parent_agent: AsyncAgent,
    spec: SubagentSpec,
    budget: SpawnBudget,
    *,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
    llm_factory: AsyncLLMFactory | None = None,
) -> AsyncAgent:
    """Build the async sub-agent for *spec* (does NOT run it).

    Mirrors :func:`~kinetic_sdk.subagent.delegation.spawn_subagent`:
    inherits tools/policy/audit/bus/hooks/classifier/context from the
    parent, gets a fresh :class:`ConversationState` with the spec's own
    system prompt, is audit-logged as ``subagent_spawn``, and publishes
    ``subagent.spawned`` on the shared bus. Never stacks guards — an
    already-guarded parent client is unwrapped first (otherwise every call
    would be charged twice and the parent's breaker would see the child's
    calls).
    """
    base_llm = parent_agent.llm
    if isinstance(base_llm, _GuardedAsyncLLMClient):
        base_llm = base_llm.inner
    if spec.model is not None and spec.model != getattr(base_llm, "model", None):
        if llm_factory is None:
            raise SubagentError(
                f"Sub-agent {spec.name!r} requests model {spec.model!r}, which "
                "differs from the parent's model; pass llm_factory to build a "
                "client for it (or leave spec.model=None to inherit)"
            )
        built = llm_factory(spec.model)
        if not isinstance(built, AsyncLLMClient):
            from kinetic_sdk.llm.async_client import SyncToAsyncLLMClient

            built = SyncToAsyncLLMClient(built)
        base_llm = built

    agent_id = uuid.uuid4().hex
    guarded_llm = _GuardedAsyncLLMClient(
        base_llm,
        budget=budget,
        breaker=RepetitionCircuitBreaker(max_consecutive_repeats),
        agent_id=agent_id,
    )

    # Imported lazily: async_tool.py imports this module (circular at top).
    from kinetic_sdk.subagent.async_tool import AsyncDelegateTool

    # File editors are cloned so every child has its own audit owner id while
    # retaining the exact same registry instance for cross-tree coordination.
    tools: list[Any] = []
    delegate_clones: list[AsyncDelegateTool] = []
    file_tool_clones: list[Any] = []
    from kinetic_sdk.files.tool import FileTool

    for tool in parent_agent._tools.values():  # noqa: SLF001 - intra-SDK access
        if isinstance(tool, AsyncDelegateTool):
            clone = tool._clone_for(budget)  # noqa: SLF001
            delegate_clones.append(clone)
            tools.append(clone)
        elif isinstance(tool, FileTool):
            file_clone = tool._clone_for()  # noqa: SLF001
            file_tool_clones.append(file_clone)
            tools.append(file_clone)
        else:
            tools.append(tool)

    sub_agent = AsyncAgent(
        llm=guarded_llm,
        tools=tools,
        state=ConversationState(system_prompt=spec.system_prompt),
        event_bus=parent_agent.event_bus,
        classifier=parent_agent.classifier,
        max_iterations=parent_agent._max_iterations_override,  # noqa: SLF001
        context_manager=parent_agent.context_manager,
        model_context_limit=parent_agent.model_context_limit,
        permission_policy=parent_agent.permission_policy,
        audit_logger=parent_agent.audit_logger,
        # The bus is shared with the parent, whose observability logger is
        # already attached — attaching again would double-log.
        observability_logger=None,
        hooks=parent_agent.hooks,
    )
    for clone in delegate_clones:
        clone.bind(sub_agent)
    for clone in file_tool_clones:
        clone.bind(sub_agent, owner_id=agent_id)

    with _AGENT_IDS_LOCK:
        _AGENT_IDS[sub_agent] = agent_id
    _audit_spawn_async(
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
                "parent_agent_id": async_agent_id_for(parent_agent),
            },
            source="subagent",
        )
    )
    return sub_agent


def _audit_spawn_async(
    parent_agent: AsyncAgent,
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
        parent_agent_id=async_agent_id_for(parent_agent),
        agent_id=agent_id,
        spec_name=spec.name,
        model=model,
        budget_used=budget.used,
        budget_max=budget.max_total_tool_calls,
    )


def _audit_finish_async(
    parent_agent: AsyncAgent,
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


async def run_async_subagent(
    parent_agent: AsyncAgent,
    spec: SubagentSpec,
    task_prompt: str,
    budget: SpawnBudget,
    *,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
    llm_factory: AsyncLLMFactory | None = None,
) -> DelegationResult:
    """Spawn the async sub-agent and await its run to completion.

    Guardrail failures propagate (``AsyncDelegateTool`` maps them to error
    results), every outcome is audit-logged as ``subagent_finished`` on the
    shared audit logger — exactly like the sync path.
    """
    sub_agent = spawn_async_subagent(
        parent_agent,
        spec,
        budget,
        max_consecutive_repeats=max_consecutive_repeats,
        llm_factory=llm_factory,
    )
    agent_id = async_agent_id_for(sub_agent)
    try:
        final_message = await sub_agent.run(task_prompt)
    except BudgetExceededError as exc:
        _audit_finish_async(parent_agent, spec, agent_id, "budget_exceeded", str(exc))
        raise
    except RepetitionLimitError as exc:
        _audit_finish_async(parent_agent, spec, agent_id, "repetition_limit", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - audit before re-raising
        _audit_finish_async(
            parent_agent, spec, agent_id, "error", f"{type(exc).__name__}: {exc}"
        )
        raise
    _audit_finish_async(parent_agent, spec, agent_id, "completed", final_message)
    return DelegationResult(final_message=final_message, agent_id=agent_id)
