"""A leak-resistant wrapper around a secret string.

:class:`SecretValue` holds a credential (API key, token, ...) while making it
hard to leak accidentally: ``repr()`` / ``str()`` always return a redacted
placeholder, so an object that ends up in a log line, an f-string, or an
exception traceback never prints the real value. The plaintext is only
available through :meth:`reveal`, which should be called at the exact point
the secret is needed (e.g. when building an HTTP ``Authorization`` header),
never earlier and never stored in a long-lived plain ``str`` variable.

This is about *lifecycle* management of secrets while the SDK runs. It
complements :func:`kinetic_sdk.security.redact.redact_secrets`, which only
scrubs secrets out of text that is *about to be logged*.
"""

from __future__ import annotations


class SecretValue:
    """Wrap a secret string so casual introspection cannot leak it.

    Guarantees:

    * ``repr()`` and ``str()`` return ``"<SecretValue: [REDACTED]>"`` and
      never contain the real value. Because ``repr`` is redacted, printing a
      container (dict, list, ``vars(obj)``) that holds a ``SecretValue`` is
      also safe.
    * :meth:`reveal` returns the real value. Call it only where the secret is
      actually consumed (building the API request), and do not keep the
      returned string around longer than necessary.
    * Equality works between two ``SecretValue`` instances holding the same
      value, so existing test logic that compares secrets keeps working.
    * ``hash()`` raises :class:`TypeError`. A value-derived hash would let an
      attacker who can observe hashes brute-force short secrets offline, and
      an ``id()``-based hash would make equal wrappers compare unequal in
      sets/dicts - both are worse than simply being unhashable. Secrets are
      looked up by key, never used as keys themselves.
    """

    __slots__ = ("_value",)

    #: Placeholder returned by ``repr``/``str``. Never contains the value.
    REDACTED_REPR = "<SecretValue: [REDACTED]>"

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"SecretValue expects a str, got {type(value).__name__}")
        self._value = value

    def reveal(self) -> str:
        """Return the real secret value.

        Call this ONLY at the point the secret is consumed (e.g. building the
        HTTP request that needs it). Do not log, store, or pass around the
        returned string.
        """
        return self._value

    def __repr__(self) -> str:
        return self.REDACTED_REPR

    def __str__(self) -> str:
        return self.REDACTED_REPR

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretValue):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        raise TypeError(
            "SecretValue is unhashable by design: a value-derived hash would "
            "make short secrets brute-forceable from observed hashes, and "
            "secrets are looked up by key, never used as dict/set keys."
        )
