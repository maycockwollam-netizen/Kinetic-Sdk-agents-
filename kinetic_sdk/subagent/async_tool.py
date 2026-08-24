"""``AsyncDelegateTool``: delegation exposed to :class:`AsyncAgent`.

Mirrors :class:`~kinetic_sdk.subagent.tool.DelegateTool` — same spec
registry validation, same bind-after-construction pattern, same clone-for-
spawn budget sharing, same secret-redaction of the child's final message —
but :meth:`execute_async` awaits the sub-agent run (the sync ``execute``
raises, matching the async-only-tool convention).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Mapping

from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.subagent.async_delegation import (
    AsyncLLMFactory,
    DelegationResult,
    run_async_subagent,
)
from kinetic_sdk.subagent.budget import (
    DEFAULT_MAX_CONSECUTIVE_REPEATS,
    SpawnBudget,
)
from kinetic_sdk.subagent.exceptions import (
    BudgetExceededError,
    RepetitionLimitError,
)
from kinetic_sdk.subagent.manifest import SubagentSpec
from kinetic_sdk.tool.base import Tool, ToolResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.async_agent import AsyncAgent

logger = logging.getLogger(__name__)

#: Same tool name as the sync variant; a mixed sync/async agent owning both
#: would collide at construction — intentional (pick one per agent).
DELEGATE_TOOL_NAME = "delegate"


class AsyncDelegateTool(Tool):
    """Spawn a registered sub-agent to work on a self-contained task (async).

    Args mirror :class:`~kinetic_sdk.subagent.tool.DelegateTool`. The tool
    must be bound to its owning agent (:meth:`bind`) before use.
    """

    name = DELEGATE_TOOL_NAME

    def __init__(
        self,
        subagents: Mapping[str, SubagentSpec] | Iterable[SubagentSpec],
        budget: SpawnBudget | None = None,
        *,
        max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
        llm_factory: AsyncLLMFactory | None = None,
    ) -> None:
        if isinstance(subagents, Mapping):
            specs = dict(subagents)
        else:
            specs = {spec.name: spec for spec in subagents}
        for key, spec in specs.items():
            if not isinstance(spec, SubagentSpec):
                raise TypeError(
                    f"AsyncDelegateTool registry values must be SubagentSpec, got "
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
        self._parent: AsyncAgent | None = None

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
    def parent(self) -> "AsyncAgent | None":
        """The agent this tool is bound to (None until :meth:`bind`)."""
        return self._parent

    def bind(self, agent: "AsyncAgent") -> None:
        """Bind the tool to its owning agent (once, after construction)."""
        self._parent = agent

    def _clone_for(self, budget: SpawnBudget) -> "AsyncDelegateTool":
        """Unbound clone sharing the spec registry but charging *budget*."""
        return AsyncDelegateTool(
            self._specs,
            budget=budget,
            max_consecutive_repeats=self._max_consecutive_repeats,
            llm_factory=self._llm_factory,
        )

    def execute(self, **_: object) -> ToolResult:
        # Async-only tool (same convention as AsyncMockTool): the async
        # agent loop always calls execute_async, so this should never fire.
        raise NotImplementedError(
            "AsyncDelegateTool is async-only; the AsyncAgent loop calls "
            "execute_async()"
        )

    async def execute_async(  # type: ignore[override]
        self, subagent_name: str, task_prompt: str
    ) -> ToolResult:
        """Spawn the named sub-agent and await its run on *task_prompt*.

        Every failure mode maps to an error :class:`ToolResult`; nothing
        raises into the parent's loop. The output carries ONLY the child's
        final message (secret-redacted) plus its audit id.
        """
        if self._parent is None:
            return ToolResult(
                error=(
                    "AsyncDelegateTool is not bound to an agent; call "
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
            result: DelegationResult = await run_async_subagent(
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
                "final_message": redact_secrets(result.final_message),
            }
        )
