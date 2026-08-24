"""Async LLM client implementations.

Two adapters ship here:

* :class:`AsyncLiteLLMClient` — the async twin of
  :class:`~kinetic_sdk.llm.client.LiteLLMClient`, driving
  ``litellm.acompletion`` (and its streaming variant) natively on the event
  loop. Message/tool translation and response parsing are reused verbatim
  from the sync client so the two backends can never drift apart.
* :class:`SyncToAsyncLLMClient` — wraps ANY synchronous
  :class:`~kinetic_sdk.llm.client.LLMClient` behind the
  :class:`~kinetic_sdk.llm.client.AsyncLLMClient` interface by running its
  calls in a worker thread. This lets :class:`AsyncAgent` consume providers
  that only ship a sync client, and is also how a scripted sync mock can be
  reused in async tests.

Like the sync client, ``litellm`` is imported lazily so the core SDK stays
zero-dependency.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncIterator

from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LiteLLMClient,
    LLMClient,
    LLMResponse,
    Message,
    StreamEvent,
    _is_retryable,
)
from kinetic_sdk.secret.value import SecretValue
from kinetic_sdk.security.redact import redact_secrets

logger = logging.getLogger(__name__)


class AsyncLiteLLMClient(AsyncLLMClient):
    """Async LLM client backed by ``litellm.acompletion``.

    Configuration and behaviour mirror
    :class:`~kinetic_sdk.llm.client.LiteLLMClient` exactly — same constructor
    arguments, same request building, same Anthropic<->OpenAI translation,
    same retry classification (408/409/429/5xx, timeouts, connection and
    rate-limit errors retried with exponential backoff + jitter; other 4xx
    raised immediately). The differences are purely mechanical:

    * ``chat`` awaits ``litellm.acompletion`` instead of calling
      ``litellm.completion``;
    * retry backoff sleeps via ``asyncio.sleep`` so the event loop is never
      blocked;
    * ``chat_stream`` is an async generator over the streaming response;
      only stream CREATION is retried — once chunks are flowing an error
      propagates rather than risking duplicated text (same rule as sync).

    ``api_key`` is stored as a :class:`~kinetic_sdk.secret.value.SecretValue`
    and only revealed when the request is built, never in ``repr`` or logs.
    """

    def __init__(
        self,
        model: str,
        api_key: str | SecretValue | None = None,
        api_base: str | None = None,
        max_tokens: int = 4096,
        timeout: float | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_delay <= 0 or retry_max_delay <= 0:
            raise ValueError("retry delays must be positive")
        self.model = model
        self.api_key = LiteLLMClient._wrap_secret(api_key)
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._litellm = LiteLLMClient._import_litellm()

    # --- public API ---------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat turn via ``litellm.acompletion``."""
        request = self._build_request(messages, tools, system, **kwargs)
        raw = await self._acompletion_with_retry(request)
        response = LiteLLMClient._parse_response(raw)
        LiteLLMClient._attach_cost(self._litellm, raw, response)
        return response

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the response as :class:`StreamEvent` deltas.

        Yields ``text`` events for each incremental chunk, then one ``done``
        event carrying the aggregated :class:`LLMResponse`. Tool calls arrive
        as per-``index`` fragments spread over chunks and are assembled once,
        at the end — identical semantics to the sync client's streamer.
        """
        request = self._build_request(messages, tools, system, **kwargs)
        request["stream"] = True
        stream = await self._acompletion_with_retry(request)
        text_buf: list[str] = []
        tool_fragments: dict[int, dict[str, Any]] = {}
        stop_reason: str | None = None
        usage: dict[str, int] = {}
        async for chunk in stream:
            choice = None
            try:
                choice = chunk.choices[0]
            except (AttributeError, IndexError):
                choice = None
            if choice is None:
                continue
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None)
            if text:
                text_buf.append(text)
                yield StreamEvent(type="text", delta=text)
            raw_calls = getattr(delta, "tool_calls", None) or []
            for rc in raw_calls:
                fn = getattr(rc, "function", None)
                if fn is None:
                    continue
                index = getattr(rc, "index", None)
                slot = tool_fragments.setdefault(
                    index if isinstance(index, int) else 0,
                    {"id": "", "name": "", "args": []},
                )
                rid = getattr(rc, "id", None)
                if rid:
                    slot["id"] = rid
                fname = getattr(fn, "name", None)
                if fname:
                    slot["name"] = fname
                frag = getattr(fn, "arguments", None)
                if frag:
                    slot["args"].append(frag)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                stop_reason = LiteLLMClient._map_stop_reason(fr)
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage.update(LiteLLMClient._parse_usage(chunk))
        tool_calls = [
            LiteLLMClient._tool_call_from_fragments(slot)
            for _, slot in sorted(tool_fragments.items())
        ]
        final = LLMResponse(
            content="".join(text_buf),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=None,
        )
        yield StreamEvent(type="done", delta=final)

    # --- internals ----------------------------------------------------

    def _build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the kwargs dict passed to ``litellm.acompletion``."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": LiteLLMClient._translate_messages(messages, system),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        translated_tools = LiteLLMClient._translate_tools(tools)
        if translated_tools is not None:
            request["tools"] = translated_tools
        if self.api_key is not None:
            # Reveal only here, at the point the real request is built.
            request["api_key"] = self.api_key.reveal()
        if self.api_base is not None:
            request["api_base"] = self.api_base
        if self.timeout is not None:
            request["timeout"] = self.timeout
        request.update(kwargs)
        return request

    async def _acompletion_with_retry(self, request: dict[str, Any]) -> Any:
        """Call ``litellm.acompletion`` with retry + backoff on transient errors."""
        attempt = 0
        while True:
            try:
                return await self._litellm.acompletion(**request)
            except Exception as exc:  # noqa: BLE001 - classified by _is_retryable
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                attempt += 1
                delay = min(
                    self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay
                )
                delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    "Async LLM request failed (attempt %d/%d), retrying in %.2fs: %s",
                    attempt,
                    self.max_retries + 1,
                    delay,
                    redact_secrets(f"{type(exc).__name__}: {exc}"),
                )
                await asyncio.sleep(delay)


class SyncToAsyncLLMClient(AsyncLLMClient):
    """Adapt any synchronous :class:`LLMClient` to the async interface.

    ``chat`` is dispatched with ``asyncio.to_thread`` so a blocking provider
    call never stalls the event loop. ``chat_stream`` collects the sync
    iterator in a thread and replays the buffered events — incremental
    delivery is lost (the sync iterator has no awaitable chunk boundary), but
    the event sequence is preserved, so an ``AsyncAgent`` streaming run still
    ends with a correct aggregated response.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        # Plain instance attribute (not a read-only property): the ABC
        # declares ``model`` as a writeable attribute, and a property
        # override is a mypy error — same rule as ``_GuardedLLMClient``.
        self.model = inner.model

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await asyncio.to_thread(
            self.inner.chat, messages, tools, system, **kwargs
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        def _collect() -> list[StreamEvent]:
            return list(self.inner.chat_stream(messages, tools, system, **kwargs))

        events = await asyncio.to_thread(_collect)
        for event in events:
            yield event
