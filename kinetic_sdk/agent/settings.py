"""Portable agent configuration that recreates an :class:`Agent` safely.

Runtime collaborators such as tools, policies and audit sinks are deliberately
injected on recreation; serialising arbitrary Python objects would be both
unsafe and non-portable.  The settings file contains only declarative knobs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.budget import RunBudget
from kinetic_sdk.llm.registry import LLMRegistry
from kinetic_sdk.tool.base import Tool


@dataclass(frozen=True)
class AgentSettings:
    """JSON-safe construction settings for one agent instance."""

    llm_profile: str
    system_prompt: str | None = None
    max_iterations: int | None = None
    model_context_limit: int | None = None
    tool_timeout: float | None = None
    validate_tool_inputs: bool = True
    parallel_tool_execution: bool = False
    memory_recall_limit: int = 3
    run_budget: dict[str, int | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.llm_profile:
            raise ValueError("llm_profile must be non-empty")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.model_context_limit is not None and self.model_context_limit < 1:
            raise ValueError("model_context_limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSettings":
        return cls(**dict(data))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "AgentSettings":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("agent settings must contain a JSON object")
        return cls.from_dict(raw)

    def create(self, registry: LLMRegistry, tools: Iterable[Tool] | None = None, **runtime: Any) -> Agent:
        """Recreate an agent, injecting non-serialisable runtime dependencies."""
        budget = RunBudget(**self.run_budget) if self.run_budget is not None else None
        agent = Agent(
            llm=registry.routed(self.llm_profile),
            tools=tools,
            max_iterations=self.max_iterations,
            model_context_limit=self.model_context_limit,
            tool_timeout=self.tool_timeout,
            validate_tool_inputs=self.validate_tool_inputs,
            parallel_tool_execution=self.parallel_tool_execution,
            memory_recall_limit=self.memory_recall_limit,
            run_budget=budget,
            **runtime,
        )
        agent.state.system_prompt = self.system_prompt
        agent.state.metadata.update(self.metadata)
        return agent
