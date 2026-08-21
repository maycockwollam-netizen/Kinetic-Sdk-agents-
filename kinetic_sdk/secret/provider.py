"""Secret providers: pluggable sources for credentials.

A :class:`SecretProvider` answers one question - "give me the value for this
key" - so the rest of the SDK never cares whether a credential came from an
environment variable, a dict injected by the host application, or (in a later
version) a cloud secret manager. Providers return plain strings (or ``None``);
wrapping into :class:`SecretValue` happens in
:class:`kinetic_sdk.secret.registry.SecretRegistry`, the single entry point
the rest of the SDK uses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class SecretProvider(ABC):
    """Interface for a source of secrets."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the secret for *key*, or ``None`` when this provider has no
        value for it. Implementations must not raise for a missing key -
        absence is signalled by returning ``None`` so a
        :class:`~kinetic_sdk.secret.registry.SecretRegistry` can fall through
        to the next provider."""


class EnvSecretProvider(SecretProvider):
    """Read secrets from process environment variables (``os.environ``).

    This is the default provider: it matches how the SDK's examples and the
    classifier (``OPENHANDS_API_KEY``) already obtain credentials.
    """

    def get(self, key: str) -> str | None:
        return os.environ.get(key)


class DictSecretProvider(SecretProvider):
    """Serve secrets from a dict supplied at construction time.

    Useful in tests (no need to mutate real environment variables) and for
    SDK users who fetch credentials from their own secret manager and want to
    inject them without touching ``os.environ``.
    """

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = dict(secrets)

    def get(self, key: str) -> str | None:
        return self._secrets.get(key)
