"""Sub-agents: delegation with inherited permissions + cost guardrails.

This is the ONLY Stage 4 module where the agent creates *more agents*
using its own tools and credentials, so it carries two risks nothing else
has: **runaway recursion burning real API budget**, and **sensitive
intermediate output leaking back into the parent's context**. The design
addresses both with three locked decisions (deliberate, not omissions):

1. **Full inheritance, no narrowing.** A spawned sub-agent receives the
   parent's ENTIRE tool set and the SAME ``permission_policy`` object
   (like the Claude Agent SDK). Hard narrowing would break the legitimate
   "narrow controller -> wider executor" pattern. The single difference
   from an exact copy: every sub-agent has its OWN ``system_prompt``
   (required in :class:`SubagentSpec`), never the parent's.
2. **No hard depth limit.** Delegation is an ordinary tool
   (:class:`DelegateTool`), so nesting is unbounded — bounded instead by
   the two guardrails below.
3. **Guardrails by budget + circuit breaker.** ONE shared
   :class:`SpawnBudget` caps total tool calls across the WHOLE tree
   (thread-safe, ready for future parallel execution); one
   :class:`RepetitionCircuitBreaker` PER AGENT trips when a single agent
   repeats the identical tool call (same name + same arguments) too many
   times consecutively. Both raise out of the sub-agent's ``run()`` and are
   converted to error ``ToolResult`` objects by ``DelegateTool`` — the
   parent's loop never crashes.

**Context isolation.** Parent -> child is exactly one ``task_prompt``
string; child -> parent is exactly the final message in
:class:`DelegationResult` (secret-redacted by ``DelegateTool``). Sub-agent
transcripts never leak into the parent's context. Every spawn and every
outcome (completed / budget_exceeded / repetition_limit / error) is
written to the shared audit log, keyed by stable agent ids, so the
delegation tree can be reconstructed after the fact.

Minimal example::

    from kinetic_sdk.subagent import DelegateTool, SpawnBudget, SubagentSpec

    specs = [
        SubagentSpec(
            name="researcher",
            system_prompt="You research one narrow question and report back.",
            description="Answers a single factual sub-question.",
        ),
    ]
    budget = SpawnBudget(max_total_tool_calls=200)
    delegate = DelegateTool(specs, budget=budget)
    root = Agent(llm=llm, tools=[delegate, *other_tools],
                 permission_policy=...)
    delegate.bind(root)
    root.run("Compare the two designs and recommend one.")
"""

from kinetic_sdk.subagent.async_delegation import (
    AsyncLLMFactory,
    async_agent_id_for,
    run_async_subagent,
    spawn_async_subagent,
)
from kinetic_sdk.subagent.async_tool import AsyncDelegateTool
from kinetic_sdk.subagent.budget import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    DEFAULT_MAX_TOTAL_TOOL_CALLS,
    RepetitionCircuitBreaker,
    SpawnBudget,
)
from kinetic_sdk.subagent.delegation import (
    DelegationResult,
    LLMFactory,
    agent_id_for,
    run_subagent,
    spawn_subagent,
)
from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
    SubagentError,
    SubagentSpecError,
)
from kinetic_sdk.subagent.manifest import (
    SUBAGENT_MAX_NAME_LENGTH,
    SUBAGENT_NAME_PATTERN,
    SubagentSpec,
)
from kinetic_sdk.subagent.tool import DELEGATE_TOOL_NAME, DelegateTool

__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_REPEATS",
    "DEFAULT_MAX_TOTAL_TOOL_CALLS",
    "DELEGATE_TOOL_NAME",
    "AsyncDelegateTool",
    "AsyncLLMFactory",
    "BudgetExceededError",
    "DelegateTool",
    "DelegationResult",
    "LLMFactory",
    "RepetitionCircuitBreaker",
    "RepetitionLimitError",
    "SpawnBudget",
    "SubagentError",
    "SubagentSpec",
    "SubagentSpecError",
    "SUBAGENT_MAX_NAME_LENGTH",
    "SUBAGENT_NAME_PATTERN",
    "agent_id_for",
    "async_agent_id_for",
    "run_async_subagent",
    "run_subagent",
    "spawn_async_subagent",
    "spawn_subagent",
]
