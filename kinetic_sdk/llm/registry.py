"""Named, serialisable LLM profiles and retryable fallback routing.

The registry deliberately stores configuration rather than live clients.  It
therefore remains stdlib-only and lets an application recreate clients at its
composition root, where credentials belong.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from kinetic_sdk.llm.client import LLMClient, LLMResponse, Message, _is_retryable


@dataclass(frozen=True)
class LLMProfile:
    """Safe-to-persist configuration for a named model endpoint.

    ``api_key`` is intentionally absent.  Store a ``secret_key`` name and
    resolve it with :mod:`kinetic_sdk.secret` when building a real client.
    """

    name: str
    model: str
    api_base: str | None = None
    max_tokens: int = 4096
    timeout: float | None = None
    secret_key: str | None = None
    fallback_profiles: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.model:
            raise ValueError("profile name and model must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fallback_profiles"] = list(self.fallback_profiles)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMProfile":
        data = dict(data)
        data["fallback_profiles"] = tuple(data.get("fallback_profiles", ()))
        return cls(**data)


class ProfileStore:
    """Atomic JSON persistence for :class:`LLMProfile` values."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, profiles: list[LLMProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {profile.name: profile.to_dict() for profile in profiles}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> list[LLMProfile]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("profile store must contain a JSON object")
        return [LLMProfile.from_dict(value) for value in raw.values()]


ClientFactory = Callable[[LLMProfile], LLMClient]


class LLMRegistry:
    """Map stable aliases to profiles and construct clients on demand."""

    def __init__(self, factory: ClientFactory) -> None:
        self._factory = factory
        self._profiles: dict[str, LLMProfile] = {}

    def register(self, profile: LLMProfile, *, replace: bool = False) -> None:
        if profile.name in self._profiles and not replace:
            raise ValueError(f"profile {profile.name!r} is already registered")
        self._profiles[profile.name] = profile

    def get(self, name: str) -> LLMProfile:
        try:
            return self._profiles[name]
        except KeyError:
            raise KeyError(f"unknown LLM profile {name!r}") from None

    def list_profiles(self) -> list[LLMProfile]:
        return list(self._profiles.values())

    def create(self, name: str) -> LLMClient:
        return self._factory(self.get(name))

    def routed(self, name: str) -> "FallbackLLMClient":
        return FallbackLLMClient(self, name)


class FallbackLLMClient(LLMClient):
    """Try a profile's transient-failure fallback chain in declaration order."""

    def __init__(self, registry: LLMRegistry, profile_name: str) -> None:
        self._registry = registry
        self._profile_name = profile_name
        self.model = profile_name  # public alias; avoid exposing provider choice.

    def _names(self) -> list[str]:
        names: list[str] = []
        current = self._profile_name
        while current not in names:
            names.append(current)
            fallbacks = self._registry.get(current).fallback_profiles
            if not fallbacks:
                break
            current = fallbacks[0]
        return names

    def chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
        system: str | None = None, **kwargs: Any,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for name in self._names():
            try:
                return self._registry.create(name).chat(messages, tools, system, **kwargs)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error
