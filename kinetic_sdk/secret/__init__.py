"""Secret package: credential lifecycle management for the SDK.

Standardises how secrets are loaded (:class:`SecretProvider` implementations),
held (:class:`SecretValue`, which never leaks via ``repr``/``str``), and
resolved (:class:`SecretRegistry`, ordered fallback across providers). This is
distinct from :mod:`kinetic_sdk.security.redact`, which only scrubs secrets
out of text about to be logged.
"""

from kinetic_sdk.secret.provider import DictSecretProvider, EnvSecretProvider, SecretProvider
from kinetic_sdk.secret.registry import SecretNotFoundError, SecretRegistry
from kinetic_sdk.secret.value import SecretValue

__all__ = [
    "DictSecretProvider",
    "EnvSecretProvider",
    "SecretNotFoundError",
    "SecretProvider",
    "SecretRegistry",
    "SecretValue",
]
