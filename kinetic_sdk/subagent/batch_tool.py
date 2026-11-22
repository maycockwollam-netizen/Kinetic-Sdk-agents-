"""``DelegateBatchTool``: hand several independent tasks to parallel sub-agents.

Claude Code-style "smooth" parallel delegation: the model passes a list of
self-contained tasks in ONE tool call and the SDK fans them out to a
:class:`~kinetic_sdk.subagent.pool.SubagentPool`, sharing one tree budget,
then returns the merged final messages. Suitable only for genuinely
independent tasks (workers must not touch the same files).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.exceptions import BudgetExceededError
from kinetic_sdk.subagent.manifest import SubagentSpec
from kinetic_sdk.subagent.pool import SubagentPool
from kinetic_sdk.tool.base import Tool, ToolResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kinetic_sdk.agent.agent import Agent

logger = logging.getLogger(__name__)

BATCH_TOOL_NAME = "delegate_batch"


class DelegateBatchTool(Tool):
    """Run several self-contained sub-agent tasks concurrently and merge.

    Args:
        subagents: Registered specs (mapping name->spec or iterable).
        budget: Shared :class:`SpawnBudget` (one for the whole tree).
        max_concurrency: Pool size (default 3).
        max_consecutive_repeats: Per-worker circuit breaker.
        llm_factory: Forwarded to the pool; gives each worker its own client.

    The tool must be bound (``bind``) to its owning agent before use.
    """

    name = BATCH_TOOL_NAME

    def __init__(
        self,
        subagents: Mapping[str, SubagentSpec] | Iterable[SubagentSpec],
        budget: SpawnBudget | None = None,
        *,
        max_concurrency: int = 3,
        max_consecutive_repeats: int = 3,
        llm_factory: Any | None = None,
    ) -> None:
        if isinstance(subagents, Mapping):
            specs = dict(subagents)
        else:
            specs = {spec.name: spec for spec in subagents}
        for key, spec in specs.items():
            if not isinstance(spec, SubagentSpec):
                raise TypeError(
                    f"DelegateBatchTool registry values must be SubagentSpec, "
                    f"got {type(spec).__name__} for {key!r}"
                )
            if key != spec.name:
                raise ValueError(f"Registry key {key!r} does not match spec name {spec.name!r}")
        self._specs = specs
        self.budget = budget if budget is not None else SpawnBudget()
        self._max_concurrency = max_concurrency
        self._max_consecutive_repeats = max_consecutive_repeats
        self._llm_factory = llm_factory
        self._parent: Agent | None = None

        catalogue = "\n".join(
            f"- {spec.name}: {spec.description or '(no description)'}"
            for spec in sorted(self._specs.values(), key=lambda s: s.name)
        )
        self.description = (
            "Delegate MULTIPLE independent tasks to registered sub-agents running "
            "in PARALLEL. Each task is self-contained (paths, constraints — workers "
            "see nothing else). Workers must NOT touch the same files. Returns the "
            "merged final messages.\n"
            f"Available sub-agents:\n{catalogue}"
        )
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of independent tasks to run in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Self-contained instructions for this worker.",
                            },
                            "subagent_name": {
                                "type": "string",
                                "enum": sorted(specs),
                            },
                        },
                        "required": ["task"],
                    },
                }
            },
            "required": ["tasks"],
        }

    def bind(self, agent: "Agent") -> None:
        """Bind this tool to its owning agent."""
        self._parent = agent

    def execute(self, tasks: list[dict[str, Any]], **_: Any) -> ToolResult:  # type: ignore[override]
        """Fan *tasks* out to the pool, then merge their final messages."""
        if not isinstance(tasks, list) or not tasks:
            return ToolResult(error="delegate_batch requires a non-empty 'tasks' list")
        if self._parent is None:
            return ToolResult(error="DelegateBatchTool is not bound to an agent; call bind() first")

        specs: list[SubagentSpec] = []
        for index, entry in enumerate(tasks):
            if not isinstance(entry, dict):
                return ToolResult(error=f"tasks[{index}] must be an object")
            task = entry.get("task")
            if not isinstance(task, str) or not task.strip():
                return ToolResult(error=f"tasks[{index}].task must be a non-empty string")
            name = entry.get("subagent_name")
            if name is None:
                if len(self._specs) == 1:
                    name = next(iter(self._specs))
                else:
                    return ToolResult(
                        error=(
                            "tasks[i].subagent_name is required when more than one "
                            "sub-agent is registered"
                        )
                    )
            if name not in self._specs:
                return ToolResult(error=f"unknown sub-agent {name!r}")
            base = self._specs[name]
            # The pool uses spec.description as the per-worker task prompt.
            specs.append(replace(base, description=task))

        try:
            pool = SubagentPool(max_concurrency=self._max_concurrency)
            results = pool.run_all(
                self._parent,
                specs,
                self.budget,
                llm_factory=self._llm_factory,
            )
        except BudgetExceededError as exc:
            return ToolResult(error=f"delegate_batch aborted: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("delegate_batch failed")
            return ToolResult(error=f"delegate_batch failed: {exc}")

        sections: list[str] = []
        for index, (entry, result) in enumerate(zip(tasks, results)):
            if result is None:
                # The pool swallowed a worker exception; keep the batch going
                # but omit the failed worker's section.
                continue
            label = entry.get("subagent_name") or f"task {index + 1}"
            sections.append(f"## {label}\n{redact_secrets(result.final_message)}")
        output = "\n\n".join(sections)
        return ToolResult(
            output=output,
            metadata={
                "count": len(results),
                "subagents": [spec.name for spec in specs],
            },
        )