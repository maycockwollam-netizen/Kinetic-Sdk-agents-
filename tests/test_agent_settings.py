from __future__ import annotations

from kinetic_sdk.agent import AgentSettings
from kinetic_sdk.llm import LLMClient, LLMProfile, LLMRegistry, LLMResponse


class _Client(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model

    def chat(self, messages, tools=None, system=None, **kwargs):
        return LLMResponse(content="ok")


def test_settings_round_trip_and_recreate_agent(tmp_path):
    settings = AgentSettings(
        llm_profile="main", system_prompt="Be precise", max_iterations=8,
        metadata={"tenant": "acme"}, run_budget={"max_llm_calls": 4, "max_total_tokens": None},
    )
    path = tmp_path / "agent.json"
    settings.save(path)
    loaded = AgentSettings.load(path)
    registry = LLMRegistry(lambda profile: _Client(profile.model))
    registry.register(LLMProfile("main", "provider/model"))
    agent = loaded.create(registry)
    assert agent.llm.model == "main"
    assert agent.state.system_prompt == "Be precise"
    assert agent.state.metadata["tenant"] == "acme"
    assert agent.max_iterations == 8
