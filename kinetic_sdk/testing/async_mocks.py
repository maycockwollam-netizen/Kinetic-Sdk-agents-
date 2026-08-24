"""Async testing fakes: drive ``AsyncAgent`` tests without network calls.

:class:`AsyncMockLLMClient` mirrors
:class:`~kinetic_sdk.testing.mocks.MockLLMClient` — the same scripted
``ResponseEntry`` list (ready-made ``LLMResponse`` or a callable
``(messages, tools, system) -> LLMResponse``), consumed one entry per turn,
with every call recorded on ``.calls``. The callable may ALSO be a
coroutine function, which is awaited. :class:`AsyncMockTool` is the async
twin of :class:`~kinetic_sdk.testing.mocks.MockTool` with a natively async
``execute_async`` (its sync ``execute`` raises — a pure-async fixture must
never be driven synchronously by accident).
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Callable, Sequence

from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LLMResponse,
    StreamEvent,
)
from kinetic_sdk.testing.mocks import ResponseEntry
from kinetic_sdk.tool.base import Tool, ToolResult

#: A scripted async response entry: a ready-made LLMResponse, a sync
#: callable, or a coroutine callable ``(messages, tools, system) ->
#: LLMResponse``.
AsyncResponseEntry = ResponseEntry | Callable[..., Any]


class AsyncMockLLMClient(AsyncLLMClient):
    """An :class:`AsyncLLMClient` replaying a pre-programmed sequence of turns.

    Args:
        responses: Entries consumed in order, one per :meth:`chat` call.
            Same script format as
            :class:`~kinetic_sdk.testing.mocks.MockLLMClient` (so one script
            can drive both sync and async agent tests), plus coroutine
            callables which are awaited. When the script runs out, a final
            empty ``end_turn`` response is returned (the loop exits).
        model: Value reported on the ``model`` attribute.

    Every call is recorded on :attr:`calls` for later assertions.
    """

    def __init__(
        self, responses: Sequence[AsyncResponseEntry], model: str = "async-mock-model"
    ) -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    #: Chunk size used by :meth:`chat_stream` when splitting a scripted
    #: response's content into text deltas (small on purpose so tests
    #: observe multiple deltas, like a real provider stream).
    STREAM_CHUNK_SIZE = 8

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._next_response(messages, tools, system, kwargs)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the next scripted response: text deltas, then ``done``.

        Consumes the script exactly like :meth:`chat` (one entry per turn),
        so the same script drives streaming and non-streaming agents. The
        final ``done`` event carries the full scripted response, tool calls
        included.
        """
        response = await self._next_response(messages, tools, system, kwargs)
        content = response.content
        for i in range(0, len(content), self.STREAM_CHUNK_SIZE):
            yield StreamEvent(type="text", delta=content[i : i + self.STREAM_CHUNK_SIZE])
        yield StreamEvent(type="done", delta=response)

    async def _next_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "tools": tools, "system": system, "kwargs": kwargs}
        )
        if not self._responses:
            return LLMResponse(content="", stop_reason="end_turn")
        entry = self._responses.pop(0)
        if callable(entry):
            result = entry(messages, tools, system)
            if inspect.isawaitable(result):
                result = await result
            return result
        return entry


class AsyncMockTool(Tool):
    """A natively async stand-in :class:`Tool`.

    Same configuration as :class:`~kinetic_sdk.testing.mocks.MockTool`:
    a fixed ``result`` (a :class:`ToolResult` verbatim, or any other value
    wrapped as output) or a ``handler(**params)`` — which may be a coroutine
    function, awaited here. ``result`` and ``handler`` are mutually
    exclusive. Every invocation is recorded on :attr:`calls`.

    The synchronous :meth:`execute` raises ``NotImplementedError``: this
    fixture exists to exercise the async tool path, and silently falling
    back to sync execution would hide wiring mistakes in tests.
    """

    def __init__(
        self,
        name: str = "async_mock_tool",
        *,
        description: str = "An async mock tool for tests.",
        parameters: dict[str, Any] | None = None,
        result: Any = None,
        handler: Callable[..., Any] | None = None,
    ) -> None:
        if result is not None and handler is not None:
            raise ValueError("Pass either a fixed result or a handler, not both.")
        self.name = name
        self.description = description
        self.parameters = parameters if parameters is not None else {
            "type": "object",
            "properties": {},
        }
        self._result = result
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def execute(self, **params: Any) -> ToolResult:
        raise NotImplementedError(
            f"{type(self).__name__} is async-only; use execute_async "
            "(drive it with AsyncAgent, not the sync Agent)"
        )

    async def execute_async(self, **params: Any) -> ToolResult:
        self.calls.append(params)
        if self._handler is not None:
            value = self._handler(**params)
            if inspect.isawaitable(value):
                value = await value
            return self._as_result(value)
        if self._result is not None:
            return self._as_result(self._result)
        return ToolResult(output={"tool": self.name, "params": params})

    @staticmethod
    def _as_result(value: Any) -> ToolResult:
        return value if isinstance(value, ToolResult) else ToolResult(output=value)
