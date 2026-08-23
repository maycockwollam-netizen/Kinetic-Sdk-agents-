"""Pluggable token counters for context-window estimation.

The zero-dependency default (:func:`~kinetic_sdk.context.manager.estimate_tokens`,
``len(text) // 4``) is calibrated for English and UNDERESTIMATES Vietnamese and
code, so the safety threshold silently shrinks for those texts. Injecting a
real tokenizer removes that bias. Counters are plain callables ``str -> int``,
so anything (tiktoken, a provider's count API, a cached heuristic) can be
injected into ``SimpleTruncateContextManager(token_counter=...)``.

``tiktoken`` stays an optional dependency: import this module freely, only
constructing :class:`TiktokenCounter` requires it (``pip install
kinetic-agent-sdk[tokens]``).
"""

from __future__ import annotations

from typing import Any


class TiktokenCounter:
    """Token counter backed by a ``tiktoken`` encoding.

    ``encoding="cl100k_base"`` is the default (GPT-4/3.5 family). It is not a
    perfect match for every provider's tokenizer, but it is far closer than
    the 4-chars-per-token heuristic for Vietnamese and code. Any encoding
    failure falls back to the heuristic instead of breaking compaction.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "TiktokenCounter requires the 'tiktoken' package. "
                "Install it with: pip install kinetic-agent-sdk[tokens]"
            ) from exc
        self.encoding = encoding
        self._tiktoken_encoding: Any = tiktoken.get_encoding(encoding)

    def __call__(self, text: str) -> int:
        from kinetic_sdk.context.manager import estimate_tokens

        try:
            return max(1, len(self._tiktoken_encoding.encode(text)))
        except Exception:  # noqa: BLE001 - never break compaction on encoding issues
            return estimate_tokens(text)
