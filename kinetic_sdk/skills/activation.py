"""Trigger-based, request-local skill catalog activation."""

from __future__ import annotations

import re

from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.skills.skill import Skill

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def rank_skills(
    skills: list[Skill],
    query: str,
    *,
    include_description: bool = False,
    limit: int | None = None,
) -> list[Skill]:
    """Return repo skills plus knowledge skills, ordered by relevance.

    Repo skills are always activated (they describe the repository the agent
    works in). Knowledge skills are selected when a *trigger* appears in
    *query* — ranked by how many distinct triggers matched (more = more
    on-topic), ties broken by name. When ``include_description`` is true, a
    knowledge skill with no trigger match is still selected if its
    description shares a token with the query (a weak signal, ranked lowest).
    ``limit`` (non-negative) caps the total returned.
    """
    message = (query or "").lower()
    query_tokens = _tokens(message)
    repo: list[Skill] = []
    scored: list[tuple[int, Skill]] = []
    for skill in skills:
        if skill.type != "knowledge":
            if skill.type == "repo":
                repo.append(skill)
            continue
        hits = skill.trigger_hits(message)
        if hits == 0 and include_description:
            if query_tokens & _tokens(skill.description):
                hits = 1  # weak, description-only match
        if hits > 0:
            scored.append((hits, skill))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    selected = repo + [skill for _hits, skill in scored]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def select_active_skills(skills: list[Skill], latest_user_message: str) -> list[Skill]:
    """Return repo skills plus knowledge skills whose trigger appears in text."""
    return rank_skills(skills, latest_user_message)


class SkillActivationHook:
    """Append a catalog of the skills relevant to each LLM request.

    The hook returns catalog text only (progressive disclosure: the model can
    then read a skill's body on demand). :meth:`rank_skills` selects and
    orders them; the catalog itself stays name-sorted for readability. The
    Agent appends it to the request's base system prompt without mutating
    persistent conversation state.

    Args:
        skills: Registry to activate from.
        include_description: Also activate knowledge skills whose *description*
            but not a trigger matches the message (weak signal, ranked last).
        max_active: Optional cap on the number of skills injected per request.
    """

    def __init__(
        self,
        skills: list[Skill],
        *,
        include_description: bool = False,
        max_active: int | None = None,
    ) -> None:
        self._skills = list(skills)
        self._include_description = include_description
        self._max_active = max_active

    def __call__(self, context: HookContext) -> HookResult | None:
        if context.point is not HookPoint.BEFORE_LLM_CALL:
            return None
        active = rank_skills(
            self._skills,
            context.user_message or "",
            include_description=self._include_description,
            limit=self._max_active,
        )
        if not active:
            return None
        catalog = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in sorted(active, key=lambda skill: skill.name)
        )
        return HookResult(modified_context={"system_prompt": catalog})
