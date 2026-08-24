"""Tests for the async LLM clients.

``AsyncLiteLLMClient`` is tested with a fake ``litellm`` module injected via
monkeypatch (same approach as ``tests/test_litellm_client.py`` — no network).
``SyncToAsyncLLMClient`` is tested against the scripted sync mock.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from kinetic_sdk.llm.async_client import AsyncLiteLLMClient, SyncToAsyncLLMClient
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.secret.value import SecretValue
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response

pytestmark = pytest.mark.asyncio


# --- fake litellm plumbing -------------------------------------------------


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments, index=None):
        self.id = id
        self.index = index
        self.function = _FakeFunction(name, arguments)


class _FakeChoice:
    def __init__(self, message=None, finish_reason=None, delta=None):
        self.message = message
        self.finish_reason = finish_reason
        self.delta = delta


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, message, finish_reason="stop", usage=None):
        self.choices = [_FakeChoice(message=message, finish_reason=finish_reason)]
        self.usage = usage if usage is not None else _FakeUsage()


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChunk:
    def __init__(self, delta, finish_reason=None, usage=None):
        self.choices = [_FakeChoice(delta=delta, finish_reason=finish_reason)]
        self.usage = usage


class FakeLitellm(types.ModuleType):
    """A stand-in ``litellm`` module with a scriptable ``acompletion``."""

    def __init__(self):
        super().__init__("litellm")
        self.handler = None  # async callable(**request) -> response
        self.requests: list[dict] = []

    async def acompletion(self, **request):
        self.requests.append(request)
        assert self.handler is not None, "no handler scripted"
        return await self.handler(**request)


@pytest.fixture
def fake_litellm(monkeypatch):
    module = FakeLitellm()
    monkeypatch.setitem(sys.modules, "litellm", module)
    return module


def make_client(**kwargs):
    kwargs.setdefault("max_retries", 0)  # tests opt in to retries explicitly
    return AsyncLiteLLMClient(model="test/model", api_key="sk-test", **kwargs)


# --- chat ------------------------------------------------------------------


async def test_chat_simple_text(fake_litellm):
    async def handler(**request):
        return _FakeResponse(_FakeMessage(content="hello there"))

    fake_litellm.handler = handler
    client = make_client()
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.content == "hello there"
    assert response.stop_reason == "end_turn"
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}
    request = fake_litellm.requests[0]
    assert request["model"] == "test/model"
    assert request["api_key"] == "sk-test"
    assert request["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_parses_tool_calls(fake_litellm):
    async def handler(**request):
        return _FakeResponse(
            _FakeMessage(
                content=None,
                tool_calls=[_FakeToolCall("call-1", "echo", '{"message": "hi"}')],
            ),
            finish_reason="tool_calls",
        )

    fake_litellm.handler = handler
    client = make_client()
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.stop_reason == "tool_use"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call-1"
    assert call.name == "echo"
    assert call.arguments == {"message": "hi"}


async def test_chat_translates_system_and_tools(fake_litellm):
    async def handler(**request):
        return _FakeResponse(_FakeMessage(content="ok"))

    fake_litellm.handler = handler
    client = make_client()
    await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        system="You are helpful.",
        tools=[{
            "name": "echo",
            "description": "Echo.",
            "input_schema": {"type": "object", "properties": {}},
        }],
    )
    request = fake_litellm.requests[0]
    assert request["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert request["tools"] == [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


async def test_chat_translates_anthropic_history(fake_litellm):
    """tool_use/tool_result blocks become OpenAI tool_calls/tool messages."""
    async def handler(**request):
        return _FakeResponse(_FakeMessage(content="done"))

    fake_litellm.handler = handler
    client = make_client()
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "c1", "name": "echo", "input": {"message": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "x"},
            ],
        },
    ]
    await client.chat(messages=history)
    messages = fake_litellm.requests[0]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "checking"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"message": "x"}
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": "x"}


async def test_api_key_stored_as_secret_value(fake_litellm):
    client = make_client()
    assert isinstance(client.api_key, SecretValue)
    assert "sk-test" not in repr(client.__dict__)


async def test_timeout_and_api_base_forwarded(fake_litellm):
    async def handler(**request):
        return _FakeResponse(_FakeMessage(content="ok"))

    fake_litellm.handler = handler
    client = make_client(api_base="https://example.test", timeout=12.5)
    await client.chat(messages=[{"role": "user", "content": "hi"}])
    request = fake_litellm.requests[0]
    assert request["api_base"] == "https://example.test"
    assert request["timeout"] == 12.5


# --- retry ------------------------------------------------------------------


class _RetryableError(Exception):
    status_code = 429


class _FatalError(Exception):
    status_code = 401


async def test_retry_on_transient_error(fake_litellm):
    attempts = 0

    async def handler(**request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _RetryableError("rate limited")
        return _FakeResponse(_FakeMessage(content="recovered"))

    fake_litellm.handler = handler
    client = make_client(max_retries=3, retry_base_delay=0.001, retry_max_delay=0.002)
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.content == "recovered"
    assert attempts == 3


async def test_no_retry_on_fatal_error(fake_litellm):
    attempts = 0

    async def handler(**request):
        nonlocal attempts
        attempts += 1
        raise _FatalError("unauthorized")

    fake_litellm.handler = handler
    client = make_client(max_retries=3, retry_base_delay=0.001)
    with pytest.raises(_FatalError):
        await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert attempts == 1


async def test_retry_exhaustion_raises(fake_litellm):
    async def handler(**request):
        raise _RetryableError("always limited")

    fake_litellm.handler = handler
    client = make_client(max_retries=2, retry_base_delay=0.001, retry_max_delay=0.002)
    with pytest.raises(_RetryableError):
        await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert len(fake_litellm.requests) == 3  # 1 initial + 2 retries


async def test_retry_backoff_does_not_block_event_loop(fake_litellm):
    attempts = 0

    async def handler(**request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _RetryableError("limited")
        return _FakeResponse(_FakeMessage(content="ok"))

    fake_litellm.handler = handler
    client = make_client(max_retries=1, retry_base_delay=0.05, retry_max_delay=0.1)
    ticked = asyncio.Event()

    async def ticker():
        await asyncio.sleep(0.01)
        ticked.set()

    helper = asyncio.create_task(ticker())
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    await helper
    assert response.content == "ok"
    assert ticked.is_set()  # the loop ran during the backoff sleep


# --- streaming ---------------------------------------------------------------


class _AsyncChunkStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def test_chat_stream_text_deltas_and_done(fake_litellm):
    async def handler(**request):
        assert request["stream"] is True
        return _AsyncChunkStream([
            _FakeChunk(_FakeDelta(content="hel")),
            _FakeChunk(_FakeDelta(content="lo")),
            _FakeChunk(_FakeDelta(content=None), finish_reason="stop"),
        ])

    fake_litellm.handler = handler
    client = make_client()
    events = [
        event
        async for event in client.chat_stream(messages=[{"role": "user", "content": "hi"}])
    ]
    texts = [e.delta for e in events if e.type == "text"]
    assert texts == ["hel", "lo"]
    done = events[-1]
    assert done.type == "done"
    assert isinstance(done.delta, LLMResponse)
    assert done.delta.content == "hello"
    assert done.delta.stop_reason == "end_turn"


async def test_chat_stream_assembles_fragmented_tool_calls(fake_litellm):
    async def handler(**request):
        return _AsyncChunkStream([
            _FakeChunk(_FakeDelta(tool_calls=[
                _FakeToolCall("c1", "echo", '{"mes', index=0),
            ])),
            _FakeChunk(_FakeDelta(tool_calls=[
                _FakeToolCall(None, None, 'sage": "hi"}', index=0),
            ])),
            _FakeChunk(_FakeDelta(), finish_reason="tool_calls"),
        ])

    fake_litellm.handler = handler
    client = make_client()
    events = [
        event
        async for event in client.chat_stream(messages=[{"role": "user", "content": "hi"}])
    ]
    done = events[-1]
    assert done.type == "done"
    assert len(done.delta.tool_calls) == 1
    call = done.delta.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "echo"
    assert call.arguments == {"message": "hi"}
    assert done.delta.stop_reason == "tool_use"


async def test_chat_stream_retries_only_stream_creation(fake_litellm):
    attempts = 0

    async def handler(**request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _RetryableError("limited")
        return _AsyncChunkStream([_FakeChunk(_FakeDelta(content="ok"), finish_reason="stop")])

    fake_litellm.handler = handler
    client = make_client(max_retries=1, retry_base_delay=0.001)
    events = [
        event
        async for event in client.chat_stream(messages=[{"role": "user", "content": "hi"}])
    ]
    assert events[-1].delta.content == "ok"
    assert attempts == 2


# --- SyncToAsyncLLMClient -----------------------------------------------------


async def test_sync_adapter_chat():
    inner = MockLLMClient([text_response("adapted")])
    client = SyncToAsyncLLMClient(inner)
    assert client.model == inner.model
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.content == "adapted"
    assert len(inner.calls) == 1


async def test_sync_adapter_chat_stream_replays_events():
    inner = MockLLMClient([text_response("streamed text here")])
    client = SyncToAsyncLLMClient(inner)
    events = [
        event
        async for event in client.chat_stream(messages=[{"role": "user", "content": "hi"}])
    ]
    texts = [e.delta for e in events if e.type == "text"]
    assert "".join(texts) == "streamed text here"
    assert events[-1].type == "done"
    assert events[-1].delta.content == "streamed text here"


async def test_sync_adapter_does_not_block_event_loop():
    import time

    def slow_handler(messages, tools, system):
        time.sleep(0.1)
        return text_response("slow")

    inner = MockLLMClient([slow_handler])
    client = SyncToAsyncLLMClient(inner)
    ticked = asyncio.Event()

    async def ticker():
        await asyncio.sleep(0.02)
        ticked.set()

    helper = asyncio.create_task(ticker())
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    await helper
    assert response.content == "slow"
    assert ticked.is_set()


async def test_sync_adapter_with_tool_calls():
    inner = MockLLMClient([tool_response("c1", "echo", {"message": "x"})])
    client = SyncToAsyncLLMClient(inner)
    response = await client.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.tool_calls[0].name == "echo"
