"""Trigger-based, request-local skill catalog activation."""

from __future__ import annotations

from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.skills.skill import Skill


def select_active_skills(skills: list[Skill], latest_user_message: str) -> list[Skill]:
    """Return repo skills plus knowledge skills whose trigger appears in text."""
    message = latest_user_message.lower()
    return [
        skill
        for skill in skills
        if skill.type == "repo"
        or (
            skill.type == "knowledge"
            and any(trigger.lower() in message for trigger in skill.triggers)
        )
    ]


class SkillActivationHook:
    """Append the catalog of skills relevant to each LLM request.

    The hook intentionally returns only catalog text. The Agent appends it to
    the request's base system prompt without mutating persistent conversation
    state.
    """

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = list(skills)

    def __call__(self, context: HookContext) -> HookResult | None:
        if context.point is not HookPoint.BEFORE_LLM_CALL:
            return None
        active = select_active_skills(self._skills, context.user_message or "")
        if not active:
            return None
        catalog = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in sorted(active, key=lambda skill: skill.name)
        )
        return HookResult(modified_context={"system_prompt": catalog})
