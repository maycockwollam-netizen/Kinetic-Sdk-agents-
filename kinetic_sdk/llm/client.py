"""Provider-agnostic LLM client interface with a LiteLLM implementation.

The agent loop talks to :class:`LLMClient` only. The default implementation,
:class:`LiteLLMClient`, is backed by the `litellm` library so a single class
can reach multiple providers (Anthropic Claude, OpenAI-compatible endpoints,
...) just by varying the ``model`` string and optional ``api_base``. This lets
the agent (Claude) and the future task classifier (a cheap OpenAI-compatible
endpoint) share one client class with different config.

Design notes:
* ``LLMClient.chat`` takes a list of messages and a list of tool definitions
  and returns an :class:`LLMResponse` describing the model's turn: text and
  any tool calls it wants the agent to run.
* The conversation history is kept in Anthropic's message format (``role`` +
  ``content`` where ``content`` is a list of typed blocks such as
  ``tool_use`` / ``tool_result``) because that is the most expressive of the
  common provider formats and is what :class:`ConversationState` stores.
  :class:`LiteLLMClient` translates that shape to/from the OpenAI message
  format that ``litellm.completion`` expects, so the agent loop and the
  conversation state are unchanged by the backend switch.
* Streaming is exposed via ``chat_stream`` which yields incremental
  :class:`StreamEvent` deltas. ``chat`` and ``chat_stream`` are independent
  code paths; a provider that does not stream can leave ``chat_stream``
  raising ``NotImplementedError``.
* ``litellm`` is an optional dependency: importing this module without it
  installed only fails when you actually instantiate :class:`LiteLLMClient`.
"""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal

from kinetic_sdk.secret.value import SecretValue
from kinetic_sdk.security.redact import redact_secrets

logger = logging.getLogger(__name__)

#: Role of a message in the conversation.
Role = Literal["system", "user", "assistant", "tool"]


def _is_retryable(exc: Exception) -> bool:
    """Classify an LLM-call failure as worth retrying (transient) or not.

    Provider-neutral: litellm raises many exception types (its own plus the
    underlying SDKs'), so we look at the HTTP ``status_code`` attribute when
    present and fall back to the exception class name. Retryable: 408, 409,
    429, any 5xx, timeouts and connection errors. Everything else - 400/401/
    403/404, invalid requests, content filters - is raised immediately.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (408, 409, 429) or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in ("timeout", "connection", "ratelimit", "rate_limit", "unavailable")
    )

#: A single message in the conversation history. The shape intentionally
#: matches Anthropic's message format (``role`` + ``content`` where ``content``
#: is a list of typed blocks) because that is the most expressive of the
#: common provider formats. Other providers translate to/from it.
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    Attributes:
        id: Provider-assigned id used to correlate the eventual tool result
            with this call.
        name: Name of the tool to invoke (must match a registered Tool).
        arguments: Parsed arguments object. Parsed from JSON where the
            provider sends a string.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """The model's turn after a ``chat`` call.

    Attributes:
        content: Full assistant text for the turn, concatenated from any
            text blocks the model produced. Empty when the turn only
            requested tool calls.
        tool_calls: Tool calls the model wants executed. Empty when the
            model is done and producing a final text answer.
        stop_reason: Provider-native stop reason (e.g. ``"end_turn"`` or
            ``"tool_use"``). Useful for debugging and tests.
        usage: Token usage info with at least ``input_tokens`` and
            ``output_tokens`` when the provider reports it, optionally a
            best-effort ``cost_usd`` float (see ``LiteLLMClient``).
        raw: The unprocessed provider response, for advanced use.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    # Values are ints for token counts, float for the optional cost.
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def reasoning_text(self) -> str | None:
        """Best-effort provider-neutral extraction of hidden reasoning text.

        Providers expose this in several incompatible shapes.  This helper is
        deliberately defensive because ``raw`` is provider-owned data.
        """
        def get(value: Any, key: str) -> Any:
            return value.get(key) if isinstance(value, dict) else getattr(value, key, None)

        def extract(value: Any) -> str | None:
            if isinstance(value, str):
                return value or None
            if isinstance(value, list):
                parts = [part for item in value if (part := extract(item))]
                return "\n".join(parts) or None
            if isinstance(value, dict) or value is not None:
                kind = get(value, "type")
                if kind in {"thinking", "reasoning"}:
                    for key in ("thinking", "text", "summary", "content"):
                        found = extract(get(value, key))
                        if found:
                            return found
                for key in ("reasoning", "reasoning_content", "thinking", "content"):
                    found = extract(get(value, key))
                    if found:
                        return found
            return None

        raw = self.raw
        direct = extract(raw)
        if direct:
            return direct
        choices = get(raw, "choices")
        if isinstance(choices, list) and choices:
            return extract(get(choices[0], "message"))
        return None


@dataclass
class StreamEvent:
    """An incremental delta from a streaming response.

    ``type`` is one of:
    - ``"text"``: a chunk of assistant text (``delta`` is the text).
    - ``"tool_call"``: a complete tool call (parsed once the provider has
      finished emitting it; streaming partial tool args is a later stage).
    - ``"done"``: the stream is complete, carries the final :class:`LLMResponse`.
    """

    type: Literal["text", "tool_call", "done"]
    delta: str | ToolCall | LLMResponse | None = None


class LLMClient(ABC):
    """Abstract interface every provider implementation must satisfy."""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a single non-streaming chat turn and return the full response.

        Args:
            messages: Conversation history (user/assistant/tool turns) in
                Anthropic message format. System prompt is passed separately.
            tools: Tool definitions (provider-neutral shape from
                :meth:`kinetic_sdk.tool.base.Tool.to_schema`) the model may
                call. ``None`` disables tool use.
            system: Optional system prompt prepended to the conversation.
            **kwargs: Provider-specific overrides (temperature, max_tokens,
                stop sequences, ...).
        """

    def count_tokens(self, text: str) -> int:
        """Return the provider's token count for *text* when it is available.

        Provider counting APIs differ: for example, some require complete
        message structures rather than a text fragment.  The base interface
        therefore deliberately remains optional.  Context counters must catch
        :class:`NotImplementedError` and use a local fallback so existing
        custom clients do not need to implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support token counting")

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Yield incremental :class:`StreamEvent` deltas.

        Default implementation raises so providers that don't stream remain
        usable via :meth:`chat` without inheriting a broken streaming API.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")


class LiteLLMClient(LLMClient):
    """LLM client backed by the `litellm` library.

    A single class reaches multiple providers: pass a LiteLLM-style ``model``
    string such as ``"anthropic/claude-sonnet-4-5"`` (Anthropic Claude) or
    ``"openai/openhands/glm-5.2"`` with a custom ``api_base`` pointing at an
    OpenAI-compatible endpoint. ``api_key`` and ``api_base`` are forwarded to
    ``litellm.completion`` on every call, so two instances of this class with
    different config act as two independent clients (the main agent model and
    the classifier model, for example).

    The conversation history is stored in Anthropic's message format (typed
    content blocks including ``tool_use`` / ``tool_result``) by
    :class:`ConversationState`. Before calling litellm we translate that to
    the OpenAI message format, and translate the OpenAI response back into the
    :class:`LLMResponse` shape the agent loop expects. This keeps the agent
    loop and conversation state provider-neutral.

    The `litellm` package is imported lazily so importing this module (and the
    rest of the SDK) does not force the dependency on environments that use a
    different provider implementation.

    ``api_key`` is stored as a :class:`~kinetic_sdk.secret.value.SecretValue`
    (plain strings are wrapped automatically, so existing callers keep
    working). The plaintext is only revealed inside ``_build_request`` when
    the actual API call is constructed - it never appears in ``repr()`` of
    the client or its ``__dict__``.

    Resilience: ``timeout`` is forwarded to ``litellm.completion`` (when set)
    and transient failures are retried with exponential backoff + jitter
    (``max_retries`` retries after the first attempt; ``0`` restores the old
    single-attempt behaviour). An error is retryable when it carries an HTTP
    status of 408/409/429 or 5xx, or its class name signals a timeout /
    connection / rate-limit problem. Anything else (4xx auth errors, invalid
    requests, ...) raises immediately - retrying those would just burn quota.
    For ``chat_stream`` only stream CREATION is retried: once chunks are
    flowing, an error propagates rather than risking duplicated text.
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
        self.api_key = self._wrap_secret(api_key)
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        # Import lazily so the rest of the SDK stays zero-dependency. We keep a
        # reference to the module to call ``completion`` on it (and so tests can
        # monkeypatch ``litellm.completion`` on the real module object).
        self._litellm = self._import_litellm()

    @staticmethod
    def _wrap_secret(api_key: str | SecretValue | None) -> SecretValue | None:
        """Normalise *api_key* to a :class:`SecretValue` (or ``None``).

        Plain strings are wrapped for backward compatibility - callers that
        still pass ``api_key="sk-..."`` work unchanged, but the key is never
        held as a bare string on the instance.
        """
        if api_key is None or isinstance(api_key, SecretValue):
            return api_key
        return SecretValue(api_key)

    @staticmethod
    def _import_litellm() -> Any:
        try:
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "LiteLLMClient requires the 'litellm' package. "
                "Install it with: pip install kinetic-agent-sdk[llm]"
            ) from exc
        return litellm

    # --- message / tool schema translation ----------------------------

    @staticmethod
    def _translate_messages(
        messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        """Translate Anthropic-format messages to the OpenAI message format.

        Anthropic stores assistant tool calls as ``tool_use`` content blocks
        and tool results as ``user`` turns containing ``tool_result`` blocks.
        OpenAI expresses the same conversation with ``assistant.tool_calls``
        and separate ``tool``-role messages keyed by ``tool_call_id``. litellm
        speaks the OpenAI shape, so we normalise here.
        """
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "assistant":
                out.append(LiteLLMClient._translate_assistant(content))
            elif role == "user" and isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    # A user turn of tool_result blocks becomes one or more
                    # OpenAI ``tool`` messages, one per result.
                    for b in tool_results:
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id", ""),
                                "content": b.get("content", ""),
                            }
                        )
                    # Any non-tool_result blocks in the same turn (rare) are
                    # appended as a follow-up user message.
                    extras = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
                    if extras:
                        out.append({"role": "user", "content": LiteLLMClient._flatten_text(extras)})
                else:
                    out.append(
                        {
                            "role": "user",
                            "content": LiteLLMClient._translate_content_blocks(content),
                        }
                    )
            else:
                # Plain text turn (user/assistant string content) or any other
                # role: pass through with stringified content.
                out.append({"role": role, "content": content if isinstance(content, str) else LiteLLMClient._flatten_text(content) if isinstance(content, list) else content})
        return out

    @staticmethod
    def _translate_content_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate Anthropic multimodal user blocks to OpenAI wire format.

        Conversation state deliberately retains Anthropic's expressive block
        format.  LiteLLM receives OpenAI-compatible messages, where base64
        images are represented by an ``image_url`` data URL instead.
        Unrecognised blocks pass through unchanged to preserve compatibility
        with future provider-supported content types.
        """
        translated: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") != "image":
                translated.append(block)
                continue

            source = block.get("source", {})
            if not isinstance(source, dict):
                source = {}
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            translated.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
        return translated

    @staticmethod
    def _translate_assistant(content: Any) -> dict[str, Any]:
        """Translate an assistant turn (string or typed blocks) to OpenAI shape."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                import json

                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    @staticmethod
    def _flatten_text(blocks: list[Any]) -> str:
        """Concatenate ``text`` blocks into a single string."""
        parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Translate Anthropic tool schemas to the OpenAI function format.

        Anthropic shape: ``{name, description, input_schema}``. OpenAI shape:
        ``{type: "function", function: {name, description, parameters}}``.
        """
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    # --- response parsing ---------------------------------------------

    @staticmethod
    def _parse_response(raw: Any) -> LLMResponse:
        """Parse a non-streaming litellm/OpenAI response into :class:`LLMResponse`."""
        import json

        choice = None
        try:
            choice = raw.choices[0]
        except (AttributeError, IndexError):
            choice = None
        message = getattr(choice, "message", None) if choice is not None else None
        content = ""
        tool_calls: list[ToolCall] = []
        if message is not None:
            text = getattr(message, "content", None)
            if text:
                content = text if isinstance(text, str) else str(text)
            raw_calls = getattr(message, "tool_calls", None) or []
            for rc in raw_calls:
                fn = getattr(rc, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                tool_calls.append(ToolCall(id=getattr(rc, "id", ""), name=name, arguments=args))
        finish = getattr(choice, "finish_reason", None) if choice is not None else None
        stop_reason = LiteLLMClient._map_stop_reason(finish)
        usage = LiteLLMClient._parse_usage(raw)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=raw,
        )

    @staticmethod
    def _map_stop_reason(finish: str | None) -> str | None:
        """Map OpenAI ``finish_reason`` to the Anthropic-style stop reason."""
        if finish is None:
            return None
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "end_turn",
        }
        return mapping.get(finish, finish)

    @staticmethod
    def _parse_usage(raw: Any) -> dict[str, int]:
        """Normalise token usage to ``{input_tokens, output_tokens}``."""
        usage = getattr(raw, "usage", None)
        if usage is None:
            return {}
        inp = getattr(usage, "prompt_tokens", None)
        out = getattr(usage, "completion_tokens", None)
        result: dict[str, int] = {}
        if inp is not None:
            result["input_tokens"] = inp
        if out is not None:
            result["output_tokens"] = out
        return result

    # --- public API ---------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-streaming chat turn via ``litellm.completion``."""
        request = self._build_request(messages, tools, system, **kwargs)
        raw = self._completion_with_retry(request)
        response = self._parse_response(raw)
        self._attach_cost(self._litellm, raw, response)
        return response

    @staticmethod
    def _attach_cost(litellm_module: Any, raw: Any, response: LLMResponse) -> None:
        """Best-effort cost lookup via ``litellm.completion_cost``.

        Fills ``response.usage["cost_usd"]`` when litellm can price the call.
        Pricing tables are optional per provider: any failure or a falsy
        value leaves the usage untouched (tokens still accumulate).
        """
        try:
            cost = litellm_module.completion_cost(completion_response=raw)
        except Exception:  # noqa: BLE001 - pricing must never break a call
            return
        if isinstance(cost, (int, float)) and cost > 0:
            response.usage["cost_usd"] = float(cost)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Stream the response as :class:`StreamEvent` deltas.

        Yields ``text`` events for each incremental text chunk, then a single
        ``done`` event carrying the aggregated :class:`LLMResponse`. Tool calls
        are emitted in the final ``done`` event (streaming partial tool
        arguments is a later stage). Falls back to ``NotImplementedError`` if
        the provider does not stream.
        """
        request = self._build_request(messages, tools, system, **kwargs)
        request["stream"] = True
        stream = self._completion_with_retry(request)
        text_buf: list[str] = []
        # OpenAI-style streaming sends a tool call as FRAGMENTS spread over
        # many chunks: the first chunk for a call carries ``index`` + ``id`` +
        # function name, later chunks at the same index carry only argument
        # text fragments. Accumulate per index and parse the concatenated
        # arguments once, at the end. Providers that send a complete call in
        # one chunk are the degenerate case (a single full fragment).
        tool_fragments: dict[int, dict[str, Any]] = {}
        stop_reason: str | None = None
        usage: dict[str, int] = {}
        for chunk in stream:
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
                stop_reason = self._map_stop_reason(fr)
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage.update(self._parse_usage(chunk))
        tool_calls = [
            self._tool_call_from_fragments(slot)
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

    @staticmethod
    def _tool_call_from_fragments(slot: dict[str, Any]) -> ToolCall:
        """Assemble one streamed tool call from its accumulated fragments."""
        raw_args = "".join(slot["args"])
        try:
            args = (
                json.loads(raw_args)
                if isinstance(raw_args, str) and raw_args
                else (raw_args or {})
            )
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        return ToolCall(id=slot["id"], name=slot["name"], arguments=args)

    # --- internals ----------------------------------------------------

    def _build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the kwargs dict passed to ``litellm.completion``."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._translate_messages(messages, system),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        translated_tools = self._translate_tools(tools)
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

    def _completion_with_retry(self, request: dict[str, Any]) -> Any:
        """Call ``litellm.completion`` with retry + backoff on transient errors."""
        attempt = 0
        while True:
            try:
                return self._litellm.completion(**request)
            except Exception as exc:  # noqa: BLE001 - classified by _is_retryable
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                attempt += 1
                delay = min(
                    self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay
                )
                delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    "LLM request failed (attempt %d/%d), retrying in %.2fs: %s",
                    attempt,
                    self.max_retries + 1,
                    delay,
                    redact_secrets(f"{type(exc).__name__}: {exc}"),
                )
                time.sleep(delay)


class AsyncLLMClient(ABC):
    """Async variant of :class:`LLMClient`, consumed by ``AsyncAgent``.

    It mirrors :class:`LLMClient` with awaitable methods. Two adapters ship
    in :mod:`kinetic_sdk.llm.async_client`: :class:`AsyncLiteLLMClient`
    (native async via ``litellm.acompletion``) and
    :class:`SyncToAsyncLLMClient` (thread-bridge over any sync client).
    """

    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming variant. Yields :class:`StreamEvent` deltas."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        # Make this a generator for type checkers even though it never runs.
        yield StreamEvent(type="done", delta=LLMResponse())  # pragma: no cover
