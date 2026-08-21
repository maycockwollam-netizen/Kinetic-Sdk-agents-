"""Secret registry: ordered resolution across multiple providers.

:class:`SecretRegistry` is the single entry point the rest of the SDK uses to
obtain credentials. It tries each registered
:class:`~kinetic_sdk.secret.provider.SecretProvider` in priority order and
returns the first hit wrapped in a
:class:`~kinetic_sdk.secret.value.SecretValue`, so callers never handle a
plaintext secret string.

Typical setup - user overrides win, environment is the fallback::

    registry = SecretRegistry([
        DictSecretProvider(user_supplied_keys),  # checked first
        EnvSecretProvider(),                     # fallback
    ])
    api_key = registry.resolve("OPENHANDS_API_KEY")  # SecretValue
"""

from __future__ import annotations

from kinetic_sdk.secret.provider import EnvSecretProvider, SecretProvider
from kinetic_sdk.secret.value import SecretValue


class SecretNotFoundError(Exception):
    """Raised when a required secret cannot be resolved from any provider.

    The message always names the missing key so SDK users immediately know
    which variable to set - unlike a bare :class:`KeyError`.
    """


class SecretRegistry:
    """Resolve secrets by trying providers in priority order.

    Args:
        providers: Providers checked in order; the first one returning a
            non-``None`` value wins. When omitted, defaults to a single
            :class:`EnvSecretProvider` (environment variables only).
    """

    def __init__(self, providers: list[SecretProvider] | None = None) -> None:
        self._providers = list(providers) if providers is not None else [EnvSecretProvider()]

    def resolve(self, key: str, required: bool = True) -> SecretValue | None:
        """Resolve *key* to a :class:`SecretValue`.

        Args:
            key: Name of the secret (e.g. an environment variable name).
            required: When ``True`` (default), raise
                :class:`SecretNotFoundError` if no provider has the key.
                When ``False``, return ``None`` instead.

        Returns:
            The resolved secret wrapped in :class:`SecretValue`, or ``None``
            when missing and ``required=False``.
        """
        for provider in self._providers:
            value = provider.get(key)
            if value is not None:
                return SecretValue(value)
        if required:
            raise SecretNotFoundError(
                f"Secret '{key}' not found in any provider. "
                f"Set the '{key}' environment variable or inject it via a "
                f"DictSecretProvider before constructing the client."
            )
        return None
