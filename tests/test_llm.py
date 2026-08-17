"""Unit tests for the LLM client interface + MockLLM used elsewhere.

We do NOT test the LiteLLM backend's network calls here; that requires the
optional ``litellm`` package and live credentials. Instead we validate the
public surface (types, no-op behaviour of the abstract methods) and the
behaviour of the :class:`MockLLM` helper used by the agent tests. The
:class:`LiteLLMClient` parsing/translation logic is unit-tested in
``tests/test_litellm_client.py`` (mocking ``litellm.completion``); the
network path is integration-tested separately when needed.
"""

from __future__ import annotations

import pytest

from kinetic_sdk.llm.client import (
    AsyncLLMClient,
    LLMClient,
    LLMResponse,
    StreamEvent,
    ToolCall,
)
from tests._helpers import MockLLM, text_response, tool_response


def test_llm_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_chat_stream_default_raises():
    class Bare(LLMClient):
        model = "bare"

        def chat(self, messages, tools=None, system=None, **kwargs):
            return LLMResponse(content="hi")

    client = Bare()
    with pytest.raises(NotImplementedError):
        next(client.chat_stream([]))


def test_tool_call_dataclass_defaults():
    call = ToolCall(id="1", name="echo")
    assert call.arguments == {}


def test_llm_response_defaults():
    resp = LLMResponse()
    assert resp.content == ""
    assert resp.tool_calls == []
    assert resp.usage == {}
    assert resp.stop_reason is None


def test_mock_llm_replays_scripted_responses():
    llm = MockLLM([text_response("hello"), tool_response("c1", "echo", {"message": "x"})])
    r1 = llm.chat(messages=[{"role": "user", "content": "hi"}])
    assert r1.content == "hello"
    r2 = llm.chat(messages=[])
    assert r2.tool_calls[0].name == "echo"
    # Exhausted script returns an empty final response.
    r3 = llm.chat(messages=[])
    assert r3.content == ""
    assert r3.tool_calls == []


def test_mock_llm_records_calls():
    llm = MockLLM([text_response("ok")])
    llm.chat(messages=[{"role": "user", "content": "q"}], tools=[{"name": "echo"}], system="sys")
    assert llm.calls[0]["system"] == "sys"
    assert llm.calls[0]["tools"] == [{"name": "echo"}]


def test_mock_llm_callable_branch():
    def branch(messages, tools, system):
        if any("hard" in str(m.get("content", "")) for m in messages):
            return tool_response("c1", "echo", {"message": "hard"})
        return text_response("easy")

    llm = MockLLM([branch, branch])
    easy = llm.chat(messages=[{"role": "user", "content": "easy task"}])
    assert easy.content == "easy"
    hard = llm.chat(messages=[{"role": "user", "content": "a hard task"}])
    assert hard.tool_calls[0].arguments["message"] == "hard"


def test_stream_event_types():
    ev = StreamEvent(type="text", delta="x")
    assert ev.type == "text"
    assert ev.delta == "x"


def test_async_llm_client_is_abstract():
    with pytest.raises(TypeError):
        AsyncLLMClient()  # type: ignore[abstract]
