"""Tests for the ``secret`` package and its wiring into the LLM client.

Covers:

* :class:`SecretValue` - redacted ``repr``/``str``, ``reveal()``, equality,
  unhashable by design.
* :class:`EnvSecretProvider` / :class:`DictSecretProvider` - basic lookup.
* :class:`SecretRegistry` - ordered resolution, ``required`` behaviour, and
  the dedicated :class:`SecretNotFoundError`.
* Integration - :class:`LiteLLMClient` and :class:`LiteLLMClassifier` accept
  both plain strings (backward compatible) and ``SecretValue``, hold the key
  wrapped (never as a bare string), and only reveal it when building the
  actual API request.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kinetic_sdk.secret import (
    DictSecretProvider,
    EnvSecretProvider,
    SecretNotFoundError,
    SecretRegistry,
    SecretValue,
)

# --- SecretValue --------------------------------------------------------


def test_secret_value_repr_and_str_never_leak():
    secret = "sk-super-secret-value-12345"
    wrapped = SecretValue(secret)
    assert repr(wrapped) == "<SecretValue: [REDACTED]>"
    assert str(wrapped) == "<SecretValue: [REDACTED]>"
    assert secret not in repr(wrapped)
    assert secret not in str(wrapped)
    # f-string / format paths go through __str__/__repr__ too
    assert secret not in f"key={wrapped}"
    assert secret not in "{}".format(wrapped)


def test_secret_value_repr_safe_inside_containers():
    secret = "sk-container-leak-check"
    wrapped = SecretValue(secret)
    assert secret not in repr({"api_key": wrapped})
    assert secret not in repr([wrapped])


def test_secret_value_reveal_returns_original():
    secret = "sk-reveal-me"
    assert SecretValue(secret).reveal() == secret


def test_secret_value_equality():
    assert SecretValue("abc") == SecretValue("abc")
    assert SecretValue("abc") != SecretValue("abd")
    assert SecretValue("abc") != "abc"  # never equal to a bare string
    assert SecretValue("abc") != 42


def test_secret_value_unhashable_by_design():
    with pytest.raises(TypeError):
        hash(SecretValue("abc"))


def test_secret_value_rejects_non_string():
    with pytest.raises(TypeError):
        SecretValue(123)  # type: ignore[arg-type]


# --- Providers ----------------------------------------------------------


def test_env_secret_provider_reads_environment(monkeypatch):
    monkeypatch.setenv("KINETIC_TEST_KEY", "from-env")
    provider = EnvSecretProvider()
    assert provider.get("KINETIC_TEST_KEY") == "from-env"


def test_env_secret_provider_missing_returns_none(monkeypatch):
    monkeypatch.delenv("KINETIC_TEST_MISSING", raising=False)
    assert EnvSecretProvider().get("KINETIC_TEST_MISSING") is None


def test_dict_secret_provider_returns_injected_values():
    provider = DictSecretProvider({"MY_KEY": "from-dict"})
    assert provider.get("MY_KEY") == "from-dict"
    assert provider.get("UNKNOWN") is None


# --- SecretRegistry -----------------------------------------------------


def test_registry_resolve_returns_secret_value():
    registry = SecretRegistry([DictSecretProvider({"K": "v"})])
    resolved = registry.resolve("K")
    assert isinstance(resolved, SecretValue)
    assert resolved.reveal() == "v"


def test_registry_resolve_missing_required_raises_with_key_name():
    registry = SecretRegistry([DictSecretProvider({})])
    with pytest.raises(SecretNotFoundError) as excinfo:
        registry.resolve("MISSING_KEY")
    assert "MISSING_KEY" in str(excinfo.value)


def test_registry_resolve_missing_optional_returns_none():
    registry = SecretRegistry([DictSecretProvider({})])
    assert registry.resolve("MISSING_KEY", required=False) is None


def test_registry_provider_priority_order():
    first = DictSecretProvider({"K": "from-first"})
    second = DictSecretProvider({"K": "from-second", "OTHER": "other"})
    registry = SecretRegistry([first, second])
    # First provider that has the key wins; later providers are not consulted.
    assert registry.resolve("K").reveal() == "from-first"
    # Fall through to the next provider when earlier ones miss the key.
    assert registry.resolve("OTHER").reveal() == "other"


def test_registry_default_uses_environment(monkeypatch):
    monkeypatch.setenv("KINETIC_DEFAULT_REG_KEY", "env-value")
    registry = SecretRegistry()
    assert registry.resolve("KINETIC_DEFAULT_REG_KEY").reveal() == "env-value"


# --- LiteLLMClient integration -------------------------------------------


def _make_text_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


@pytest.fixture()
def fake_completion(monkeypatch):
    import litellm

    calls: list[dict[str, Any]] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _make_text_response("ok")

    _completion.calls = calls
    monkeypatch.setattr(litellm, "completion", _completion)
    return _completion


def test_litellm_client_accepts_plain_string_backward_compatible(fake_completion):
    from kinetic_sdk.llm.client import LiteLLMClient

    client = LiteLLMClient(model="openai/openhands/glm-5.2", api_key="sk-plain")
    # Stored wrapped, never as a bare string...
    assert isinstance(client.api_key, SecretValue)
    # ...but the outgoing request still carries the real key.
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert fake_completion.calls[0]["api_key"] == "sk-plain"


def test_litellm_client_accepts_secret_value(fake_completion):
    from kinetic_sdk.llm.client import LiteLLMClient

    client = LiteLLMClient(
        model="openai/openhands/glm-5.2", api_key=SecretValue("sk-wrapped")
    )
    assert client.api_key is not None
    assert client.api_key.reveal() == "sk-wrapped"
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert fake_completion.calls[0]["api_key"] == "sk-wrapped"


def test_litellm_client_introspection_does_not_leak_key():
    from kinetic_sdk.llm.client import LiteLLMClient

    secret = "sk-do-not-leak-me"
    client = LiteLLMClient(model="openai/openhands/glm-5.2", api_key=secret)
    assert secret not in repr(client)
    # Dumping the instance state (e.g. logging vars(obj)) must stay safe.
    assert secret not in repr(vars(client))
    assert secret not in json.dumps(repr(vars(client)))


def test_litellm_client_without_key_stays_none(fake_completion):
    from kinetic_sdk.llm.client import LiteLLMClient

    client = LiteLLMClient(model="openai/openhands/glm-5.2")
    assert client.api_key is None
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert "api_key" not in fake_completion.calls[0]


# --- LiteLLMClassifier integration ---------------------------------------


class _StubClient:
    """Minimal stand-in for the classifier's LLM client (no network)."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def chat(self, messages, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content="COMPLEX")


def test_classifier_wraps_plain_string_key():
    from kinetic_sdk.agent.classifier import LiteLLMClassifier

    clf = LiteLLMClassifier(client=_StubClient(), api_key="sk-clf")
    assert isinstance(clf._api_key, SecretValue)
    assert clf._api_key.reveal() == "sk-clf"
    assert "sk-clf" not in repr(vars(clf))


def test_classifier_accepts_secret_value():
    from kinetic_sdk.agent.classifier import LiteLLMClassifier

    wrapped = SecretValue("sk-clf-wrapped")
    clf = LiteLLMClassifier(client=_StubClient(), api_key=wrapped)
    assert clf._api_key is wrapped


def test_classifier_resolves_key_from_env_by_default(monkeypatch):
    from kinetic_sdk.agent.classifier import LiteLLMClassifier

    monkeypatch.setenv("OPENHANDS_API_KEY", "sk-from-env")
    clf = LiteLLMClassifier(client=_StubClient())
    assert isinstance(clf._api_key, SecretValue)
    assert clf._api_key.reveal() == "sk-from-env"


def test_classifier_resolves_key_from_custom_registry():
    from kinetic_sdk.agent.classifier import LiteLLMClassifier

    registry = SecretRegistry([DictSecretProvider({"OPENHANDS_API_KEY": "sk-injected"})])
    clf = LiteLLMClassifier(client=_StubClient(), secrets=registry)
    assert clf._api_key.reveal() == "sk-injected"


def test_classifier_missing_key_resolves_to_none(monkeypatch):
    from kinetic_sdk.agent.classifier import LiteLLMClassifier

    monkeypatch.delenv("OPENHANDS_API_KEY", raising=False)
    clf = LiteLLMClassifier(client=_StubClient())
    assert clf._api_key is None
