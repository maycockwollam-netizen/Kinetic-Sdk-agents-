"""``DelegateTool``: delegation exposed as an ordinary tool.

True to the "delegation is a tool, not special core" philosophy, the parent
agent delegates by calling this tool through the regular tool-call path —
meaning the call itself passes through ``permission_policy.check`` like any
other tool (deny ``"delegate"`` in an :class:`AllowListPolicy` and no
sub-agent ever spawns). There is no special branch in ``Agent.run`` for it.

Usage::

    budget = SpawnBudget()                     # one budget per root task
    delegate = DelegateTool([spec_a, spec_b], budget=budget)
    root = Agent(llm=..., tools=[delegate, ...], permission_policy=...)
    delegate.bind(root)                        # bind AFTER constructing root
    root.run("...")

When a sub-agent is spawned, every ``DelegateTool`` in the parent's tool
set is cloned and re-bound to the sub-agent (same spec registry, SAME
budget instance) — that is how multi-level trees share one budget.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Mapping

from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.subagent.budget import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    SpawnBudget,
)
from kinetic_sdk.subagent.delegation import (
    DelegationResult,
    LLMFactory,
    run_subagent,
)
from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
)
from kinetic_sdk.subagent.manifest import SubagentSpec
from kinetic_sdk.tool.base import Tool, ToolResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.agent import Agent

logger = logging.getLogger(__name__)

#: Tool name the model calls. A parent that also registers a different tool
#: under this name raises on duplicate at Agent construction — intentional.
DELEGATE_TOOL_NAME = "delegate"


class DelegateTool(Tool):
    """Spawn a registered sub-agent to work on a self-contained task.

    Args:
        subagents: The registry of spawnable sub-agents — a mapping of name
            to :class:`SubagentSpec`, or an iterable of specs (keyed by
            their ``name``). Snapshotted at construction.
        budget: Shared :class:`SpawnBudget` for the whole tree. ``None``
            creates a fresh one with the default cap — pass an explicit
            instance whenever the caller needs to observe/enforce it.
        max_consecutive_repeats: Circuit-breaker limit applied to each
            sub-agent this tool spawns.
        llm_factory: Forwarded to
            :func:`~kinetic_sdk.subagent.delegation.spawn_subagent`. When
            provided, it builds the child client even for an inherited model.

    The tool must be bound to its owning agent (:meth:`bind`) before use;
    :func:`~kinetic_sdk.subagent.delegation.spawn_subagent` clones and
    re-binds it automatically for spawned sub-agents.
    """

    name = DELEGATE_TOOL_NAME

    def __init__(
        self,
        subagents: Mapping[str, SubagentSpec] | Iterable[SubagentSpec],
        budget: SpawnBudget | None = None,
        *,
        max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        if isinstance(subagents, Mapping):
            specs = dict(subagents)
        else:
            specs = {spec.name: spec for spec in subagents}
        for key, spec in specs.items():
            if not isinstance(spec, SubagentSpec):
                raise TypeError(
                    f"DelegateTool registry values must be SubagentSpec, got "
                    f"{type(spec).__name__} for {key!r}"
                )
            if key != spec.name:
                raise ValueError(
                    f"Registry key {key!r} does not match spec name {spec.name!r}"
                )
        self._specs = specs
        self.budget = budget if budget is not None else SpawnBudget()
        self._max_consecutive_repeats = max_consecutive_repeats
        self._llm_factory = llm_factory
        self._parent: Agent | None = None

        catalogue = "\n".join(
            f"- {spec.name}: {spec.description or '(no description)'}"
            for spec in sorted(self._specs.values(), key=lambda s: s.name)
        )
        self.description = (
            "Delegate a self-contained task to a registered sub-agent. The "
            "sub-agent inherits your tools and permissions, starts from a "
            "fresh context with its own system prompt, and returns only its "
            "final message — everything it needs must fit in task_prompt. "
            f"Available sub-agents:\n{catalogue}"
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "subagent_name": {
                    "type": "string",
                    "description": "Name of the registered sub-agent to spawn.",
                    "enum": sorted(self._specs),
                },
                "task_prompt": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained instructions for the "
                        "sub-agent (file paths, decisions, constraints — the "
                        "sub-agent sees nothing else from your context)."
                    ),
                },
            },
            "required": ["subagent_name", "task_prompt"],
        }

    @property
    def subagent_specs(self) -> dict[str, SubagentSpec]:
        """A copy of the registered specs (name -> spec)."""
        return dict(self._specs)

    @property
    def parent(self) -> "Agent | None":
        """The agent this tool is bound to (None until :meth:`bind`)."""
        return self._parent

    def bind(self, agent: "Agent") -> None:
        """Bind the tool to its owning agent (once, after construction)."""
        self._parent = agent

    def _clone_for(self, budget: SpawnBudget) -> "DelegateTool":
        """Unbound clone sharing the spec registry but charging *budget*.

        Used by ``spawn_subagent`` so nested delegations draw from the same
        shared tree budget instead of starting a fresh one.
        """
        return DelegateTool(
            self._specs,
            budget=budget,
            max_consecutive_repeats=self._max_consecutive_repeats,
            llm_factory=self._llm_factory,
        )

    # Named-parameter signature by design: the agent loop always invokes
    # tools via execute(**model_arguments), so narrowing **params is safe.
    def execute(  # type: ignore[override]
        self, subagent_name: str, task_prompt: str
    ) -> ToolResult:
        """Spawn the named sub-agent and run it on *task_prompt*.

        Every failure mode — unknown name, unbound tool, exhausted budget,
        tripped circuit breaker, sub-agent crash — is returned as an error
        :class:`ToolResult`; nothing raises into the parent's agent loop.
        The output carries ONLY the sub-agent's final message (secret-
        redacted) plus its audit id — never the transcript.
        """
        if self._parent is None:
            return ToolResult(
                error=(
                    "DelegateTool is not bound to an agent; call "
                    "delegate_tool.bind(agent) after constructing the agent"
                )
            )
        spec = self._specs.get(subagent_name)
        if spec is None:
            available = ", ".join(sorted(self._specs)) or "(none)"
            return ToolResult(
                error=f"Unknown sub-agent {subagent_name!r}. Registered: {available}"
            )
        try:
            result: DelegationResult = run_subagent(
                self._parent,
                spec,
                task_prompt,
                self.budget,
                max_consecutive_repeats=self._max_consecutive_repeats,
                llm_factory=self._llm_factory,
            )
        except BudgetExceededError as exc:
            logger.warning("Delegation to %r blocked: %s", spec.name, exc)
            return ToolResult(error=f"BudgetExceededError: {exc}")
        except RepetitionLimitError as exc:
            logger.warning("Delegation to %r blocked: %s", spec.name, exc)
            return ToolResult(error=f"RepetitionLimitError: {exc}")
        except Exception as exc:  # noqa: BLE001 - never sink the parent's loop
            logger.exception("Delegation to %r failed", spec.name)
            return ToolResult(error=f"{type(exc).__name__}: {exc}")
        return ToolResult(
            output={
                "subagent": spec.name,
                "agent_id": result.agent_id,
                # Redacted before entering the parent's context: the
                # sub-agent's final message may quote secrets it handled.
                "final_message": redact_secrets(result.final_message),
            }
        )
