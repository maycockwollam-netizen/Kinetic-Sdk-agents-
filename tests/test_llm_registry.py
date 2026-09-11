from __future__ import annotations

from kinetic_sdk.llm import (
    LLMClient,
    LLMProfile,
    LLMRegistry,
    LLMResponse,
    ProfileStore,
)


class Client(LLMClient):
    def __init__(self, model: str, response: LLMResponse | Exception) -> None:
        self.model = model
        self.response = response
        self.calls = 0

    def chat(self, messages, tools=None, system=None, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TimeoutError(Exception):
    pass


def test_profile_store_round_trip_is_atomic_and_never_contains_api_key(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    profile = LLMProfile("fast", "provider/model", secret_key="FAST_KEY", fallback_profiles=("safe",))
    store.save([profile])
    assert store.load() == [profile]
    assert "api_key" not in store.path.read_text()


def test_registry_rejects_duplicate_and_creates_client():
    registry = LLMRegistry(lambda profile: Client(profile.model, LLMResponse(content="ok")))
    registry.register(LLMProfile("main", "model"))
    assert registry.create("main").chat([]).content == "ok"
    try:
        registry.register(LLMProfile("main", "other"))
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate profile was accepted")


def test_routed_client_falls_back_only_for_transient_errors():
    created = {}
    def factory(profile):
        response = TimeoutError("retry") if profile.name == "fast" else LLMResponse(content="safe")
        client = Client(profile.model, response)
        created[profile.name] = client
        return client
    registry = LLMRegistry(factory)
    registry.register(LLMProfile("fast", "fast-model", fallback_profiles=("safe",)))
    registry.register(LLMProfile("safe", "safe-model"))
    assert registry.routed("fast").chat([]).content == "safe"
    assert created["fast"].calls == created["safe"].calls == 1


def test_route_uses_dynamic_candidates_after_declared_chain():
    clients = {}
    def factory(profile):
        client = Client(profile.model, TimeoutError("retry") if profile.name != "last" else LLMResponse(content="ok"))
        clients[profile.name] = client
        return client
    registry = LLMRegistry(factory)
    registry.register(LLMProfile("first", "one", fallback_profiles=("second",)))
    registry.register(LLMProfile("second", "two"))
    registry.register(LLMProfile("last", "three"))
    assert registry.route(["first", "last"]).chat([]).content == "ok"
    assert set(clients) == {"first", "second", "last"}
