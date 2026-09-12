"""Run with ``python examples/llm_fallback.py`` to demonstrate LLM profile fallback."""

from kinetic_sdk.llm import LLMProfile, LLMRegistry
from kinetic_sdk.testing import MockLLMClient, text_response


def make_client(profile: LLMProfile):
    if profile.name == "primary":
        class TemporaryError(Exception):
            status_code = 503
        raise TemporaryError("primary unavailable")
    return MockLLMClient([text_response("fallback reply")])


if __name__ == "__main__":
    registry = LLMRegistry(make_client)
    registry.register(LLMProfile("primary", "demo", fallback_profiles=("backup",)))
    registry.register(LLMProfile("backup", "demo"))
    print(registry.routed("primary").chat([{"role": "user", "content": "hello"}]).content)
