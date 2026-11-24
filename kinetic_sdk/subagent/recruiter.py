"""Natural-language routing to a sub-agent: pick a spec, then run it.

``SubagentOrchestrator`` routes work items that already name a spec. This
module closes the gap: you hand it a NATURAL-LANGUAGE task and a registry of
``SubagentSpec`` instances; :func:`route_subagent` asks an LLM to choose the
most suitable spec (from the ``name: description`` catalogue) AND to rewrite
the task into a self-contained prompt for that spec; :func:`delegate_naturally`
then runs it. The route decision is a single cheap LLM call — the delegation
itself still goes through the normal, isolated ``run_subagent`` path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.subagent.budget import SpawnBudget
from kinetic_sdk.subagent.delegation import DelegationResult, run_subagent
from kinetic_sdk.subagent.exceptions import SubagentError
from kinetic_sdk.subagent.manifest import SubagentSpec

#: Fixed output contract the router must satisfy.
ROUTING_SYSTEM_PROMPT = (
    "You route a request to the single most suitable sub-agent. Reply with "
    "ONLY a JSON object of the form "
    '{"subagent": "<name>", "task_prompt": "<the task rewritten to be '
    'self-contained for that sub-agent>"}. The task_prompt must carry every '
    "constraint the sub-agent needs, since it sees nothing else."
)


def _parse_routing(content: str) -> dict[str, Any]:
    """Best-effort parse of the router reply; tolerate code-fenced JSON."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubagentError(f"router did not return JSON: {content[:200]!r}") from exc
    if not isinstance(data, dict):
        raise SubagentError(f"router returned a non-object: {data!r}")
    return data


def route_subagent(
    specs: Mapping[str, SubagentSpec],
    task: str,
    llm: LLMClient,
) -> tuple[SubagentSpec, str]:
    """Pick the best *spec* for *task* via an LLM, and specialise the prompt.

    Returns ``(spec, task_prompt)`` where ``task_prompt`` is the router's
    self-contained rewrite (falling back to the original *task* when the
    router did not provide one). Raises :class:`SubagentError` if the LLM
    reply cannot be parsed or names an unknown sub-agent.
    """
    catalogue = "\n".join(
        f"- {name}: {spec.description or '(no description)'}"
        for name, spec in sorted(specs.items())
    )
    user = f"Available sub-agents:\n{catalogue}\n\nTask:\n{task}"
    response = llm.chat(
        [{"role": "user", "content": user}],
        system=ROUTING_SYSTEM_PROMPT,
        max_tokens=400,
    )
    data = _parse_routing(response.content)
    name = data.get("subagent")
    if not isinstance(name, str) or name not in specs:
        raise SubagentError(
            f"router selected unknown sub-agent {name!r} (known: "
            f"{', '.join(sorted(specs)) or 'none'})"
        )
    task_prompt = data.get("task_prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        task_prompt = task
    return specs[name], task_prompt


def delegate_naturally(
    parent: Agent,
    specs: Mapping[str, SubagentSpec],
    task: str,
    *,
    budget: SpawnBudget | None = None,
    llm: LLMClient | None = None,
    llm_factory: Callable[[str], Any] | None = None,
) -> DelegationResult:
    """Route *task* to the best sub-agent and run it (one-shot dispatch).

    ``llm`` defaults to ``parent.llm`` for the (single) routing call. The
    actual delegation inherits the parent's tools/policy and shares
    *budget*, exactly like any other ``run_subagent``.
    """
    if not specs:
        raise SubagentError("no sub-agents registered to route to")
    spec, task_prompt = route_subagent(specs, task, llm if llm is not None else parent.llm)
    return run_subagent(
        parent, spec, task_prompt, budget or SpawnBudget(), llm_factory=llm_factory
    )


def delegate_by_complexity(
    parent: Agent,
    tiers: Mapping[str, Mapping[str, SubagentSpec]],
    task: str,
    *,
    budget: SpawnBudget | None = None,
    classifier: Any | None = None,
    llm: LLMClient | None = None,
    llm_factory: Callable[[str], Any] | None = None,
) -> DelegationResult:
    """Route *task* to a cheap or deep tier based on complexity, then delegate.

    ``tiers`` maps ``"flash"`` / ``"max"`` to spec registries. The task is
    classified with the parent's classifier (or *classifier*) and routed
    within the matching tier; a classifier that fails, or is absent, falls
    back to the ``"max"`` tier (the safe side). If a tier is empty, the other
    is used. See :func:`delegate_naturally` for how the per-tier route runs.
    """
    if not tiers:
        raise SubagentError("no sub-agent tiers registered to route to")
    decision = classifier if classifier is not None else getattr(parent, "classifier", None)
    prefer_max = True
    if decision is not None:
        try:
            preference = decision.classify(task)
            prefer_max = getattr(preference, "mode", AgentMode.MAX) is not AgentMode.FLASH
        except Exception:  # noqa: BLE001 - classifier failure -> safe MAX
            prefer_max = True
    candidates = tiers.get("max") if prefer_max else tiers.get("flash")
    if not candidates:
        candidates = tiers.get("flash") or tiers.get("max") or {}
    return delegate_naturally(
        parent,
        candidates,
        task,
        budget=budget,
        llm=llm,
        llm_factory=llm_factory,
    )