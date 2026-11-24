"""Tests for subagent/recruiter.py — route_subagent + delegate_naturally."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.subagent import (
    SpawnBudget,
    SubagentError,
    SubagentSpec,
    delegate_naturally,
    route_subagent,
)
from kinetic_sdk.testing import MockLLMClient, text_response


def _specs() -> dict[str, SubagentSpec]:
    return {
        "writer": SubagentSpec(name="writer", system_prompt="p", description="writes essays and prose"),
        "coder": SubagentSpec(name="coder", system_prompt="p", description="writes and fixes code"),
    }


def test_router_catalogue_includes_boundaries() -> None:
    def route_mock(messages, tools, system):  # type: ignore[no-untyped-def]
        return text_response('{"subagent": "coder", "task_prompt": "x"}')

    captured: dict[str, str] = {}

    class Recording(MockLLMClient):
        def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[override]
            captured["user"] = messages[0]["content"]
            return route_mock(messages, tools, system)

    specs = {
        "coder": SubagentSpec(
            name="coder",
            system_prompt="p",
            description="fixes code",
            boundaries=("no frontend", "no design"),
        )
    }
    route_subagent(specs, "fix a bug", Recording([]))
    assert "no frontend" in captured["user"]
    assert "no design" in captured["user"]


def test_route_subagent_selects_by_description() -> None:
    route_llm = MockLLMClient(
        [text_response('{"subagent": "coder", "task_prompt": "implement the sort function"}')]
    )
    spec, task_prompt = route_subagent(_specs(), "implement a sorting function", route_llm)
    assert spec.name == "coder"
    assert task_prompt == "implement the sort function"


def test_route_subagent_falls_back_to_original_task() -> None:
    route_llm = MockLLMClient([text_response('{"subagent": "writer", "task_prompt": ""}')])
    spec, task_prompt = route_subagent(_specs(), "write an essay about rain", route_llm)
    assert spec.name == "writer"
    assert task_prompt == "write an essay about rain"


def test_route_subagent_unknown_name_or_garbage_raises() -> None:
    bad = MockLLMClient([text_response('{"subagent": "nope", "task_prompt": "x"}')])
    with pytest.raises(SubagentError, match="unknown sub-agent"):
        route_subagent(_specs(), "t", bad)
    garbage = MockLLMClient([text_response("just text, not json")])
    with pytest.raises(SubagentError, match="did not return JSON"):
        route_subagent(_specs(), "t", garbage)


def test_delegate_naturally_routes_then_runs() -> None:
    # One routing turn (its reply content is what the router parses) then the
    # delegated child's reply.
    llm = MockLLMClient([
        text_response('{"subagent": "coder", "task_prompt": "SPECIAL X"}'),
        text_response("child finished"),
    ])
    parent = Agent(llm=llm, permission_policy=PermissivePolicy())
    result = delegate_naturally(parent, _specs(), "make a function", budget=SpawnBudget(100))
    assert result.final_message == "child finished"
    assert result.status.value == "success"


def _decision(mode) -> object:
    class Decision:
        pass

    d = Decision()
    d.mode = mode
    return d


def _flash_classifier():
    from kinetic_sdk.agent.modes import AgentMode

    class C:
        def classify(self, task):
            return _decision(AgentMode.FLASH)

    return C()


def test_delegate_by_complexity_picks_tier_by_classifier() -> None:
    from kinetic_sdk.subagent import delegate_by_complexity

    flash = SubagentSpec(name="fast", system_prompt="p", description="fast agent")
    deep = SubagentSpec(name="deep", system_prompt="p", description="deep agent")
    tiers = {"flash": {"fast": flash}, "max": {"deep": deep}}
    # Flash classification -> routes within the "flash" tier (names the spec "fast").
    llm = MockLLMClient([
        text_response('{"subagent": "fast", "task_prompt": "QUICK"}'),
        text_response("fast done"),
    ])
    parent = Agent(llm=llm, permission_policy=PermissivePolicy())
    result = delegate_by_complexity(
        parent, tiers, "a quick tweak", budget=SpawnBudget(100), classifier=_flash_classifier()
    )
    assert result.final_message == "fast done"


def test_delegate_by_complexity_falls_back_to_max_on_classifier_failure() -> None:
    from kinetic_sdk.subagent import delegate_by_complexity

    flash = SubagentSpec(name="fast", system_prompt="p", description="fast agent")
    deep = SubagentSpec(name="deep", system_prompt="p", description="deep agent")
    tiers = {"flash": {"fast": flash}, "max": {"deep": deep}}

    class Boom:
        def classify(self, task):
            raise RuntimeError("classifier offline")

    llm = MockLLMClient([
        text_response('{"subagent": "deep", "task_prompt": "DEEP"}'),
        text_response("deep done"),
    ])
    parent = Agent(llm=llm, permission_policy=PermissivePolicy())
    result = delegate_by_complexity(
        parent, tiers, "hard task", budget=SpawnBudget(100), classifier=Boom()
    )
    assert result.final_message == "deep done"


def test_delegate_naturally_rejects_empty_registry() -> None:
    parent = Agent(llm=MockLLMClient([]), permission_policy=PermissivePolicy())
    with pytest.raises(SubagentError, match="no sub-agents"):
        delegate_naturally(parent, {}, "t")