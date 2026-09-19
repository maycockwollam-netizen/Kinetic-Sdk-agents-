"""Pluggable token counters for context-window estimation.

The zero-dependency default (:func:`~kinetic_sdk.context.manager.estimate_tokens`,
``len(text) // 4``) is calibrated for English and UNDERESTIMATES Vietnamese and
code. Counters are plain callables ``str -> int`` and can be injected into a
context manager.

``tiktoken`` remains optional. :class:`ProviderTokenCounter` additionally lets
an application use a provider-specific ``LLMClient.count_tokens`` implementation
when it has one; failures and unsupported clients safely use its local fallback.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from kinetic_sdk.llm.client import LLMClient


class ProviderTokenCounter:
    """Count via an optional provider method, with cached safe fallback.

    The provider call is deliberately attempted only once per distinct text for
    this counter instance. This avoids repeated network round-trips while a
    context manager evaluates the same history during a run. Any error,
    including an unsupported :meth:`LLMClient.count_tokens`, falls back without
    letting token accounting break compaction.
    """

    def __init__(
        self,
        llm: LLMClient,
        fallback: Callable[[str], int] | None = None,
        max_cache_entries: int = 4_096,
    ) -> None:
        if max_cache_entries < 1:
            raise ValueError("max_cache_entries must be >= 1")
        self._llm = llm
        self._fallback = fallback or self._default_fallback
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _default_fallback(text: str) -> int:
        # Local import avoids a manager <-> tokens import cycle.
        from kinetic_sdk.context.manager import estimate_tokens

        return estimate_tokens(text)

    def __call__(self, text: str) -> int:
        # Do not retain arbitrary conversation/tool-output text in memory.
        key = sha1(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            count = self._llm.count_tokens(text)
            if not isinstance(count, int) or count < 0:
                raise ValueError("provider returned an invalid token count")
        except Exception:  # noqa: BLE001 - token counting must never break compaction
            count = self._fallback(text)
        result = max(1, count)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return result


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
