"""Unit tests for :class:`LiteLLMClient`.

These do NOT hit the network. We monkeypatch ``litellm.completion`` on the
real litellm module (which the client imports lazily) with fakes that return
OpenAI-shaped response objects, then assert:

* the OpenAI response is parsed into :class:`LLMResponse` / :class:`ToolCall`,
* Anthropic-format messages and tool schemas are translated to OpenAI shape,
* ``api_key`` / ``api_base`` / ``model`` are forwarded to litellm,
* streaming aggregates text deltas and emits a ``done`` event,
* the ``LLMClient`` ABC surface is unchanged (the agent tests depend on it).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kinetic_sdk.llm.client import LLMClient, LiteLLMClient, LLMResponse, ToolCall


def _make_text_response(text: str, finish: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _make_tool_response(
    call_id: str, name: str, arguments: dict[str, Any], finish: str = "tool_calls"
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
                        )
                    ],
                ),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
    )


@pytest.fixture()
def fake_completion(monkeypatch):
    """Monkeypatch ``litellm.completion`` and record the call kwargs."""
    import litellm

    calls: list[dict[str, Any]] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _completion.return_value

    _completion.return_value = None
    _completion.calls = calls
    monkeypatch.setattr(litellm, "completion", _completion)
    return _completion


def _new_client(**kwargs):
    return LiteLLMClient(model="openai/openhands/glm-5.2", **kwargs)


def test_litellm_client_is_llm_client():
    client = _new_client()
    assert isinstance(client, LLMClient)
    assert client.model == "openai/openhands/glm-5.2"


def test_chat_parses_text_response(fake_completion):
    fake_completion.return_value = _make_text_response("hello world")
    client = _new_client()
    resp = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, LLMResponse)
    assert resp.content == "hello world"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5}


def test_chat_parses_tool_calls(fake_completion):
    fake_completion.return_value = _make_tool_response("c1", "echo", {"message": "x"})
    client = _new_client()
    resp = client.chat(messages=[{"role": "user", "content": "echo x"}])
    assert resp.content == ""
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.id == "c1"
    assert call.name == "echo"
    assert call.arguments == {"message": "x"}
    assert resp.stop_reason == "tool_use"


def test_chat_translates_tool_schema_to_openai(fake_completion):
    fake_completion.return_value = _make_text_response("ok")
    client = _new_client()
    anthropic_tools = [
        {"name": "echo", "description": "Echo input.", "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}}}
    ]
    client.chat(messages=[{"role": "user", "content": "hi"}], tools=anthropic_tools)
    sent = fake_completion.calls[0]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo input.",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        }
    ]


def test_chat_translates_anthropic_messages_to_openai(fake_completion):
    """Assistant tool_use -> assistant.tool_calls; tool_result -> tool role."""
    fake_completion.return_value = _make_text_response("done")
    client = _new_client()
    messages = [
        {"role": "user", "content": "please echo"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling echo"},
                {"type": "tool_use", "id": "c1", "name": "echo", "input": {"message": "hi"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "hi"}]},
    ]
    client.chat(messages=messages, system="be helpful")
    sent = fake_completion.calls[0]["messages"]
    # system prompt prepended
    assert sent[0] == {"role": "system", "content": "be helpful"}
    assert sent[1] == {"role": "user", "content": "please echo"}
    # assistant turn with tool_calls
    assert sent[2]["role"] == "assistant"
    assert sent[2]["content"] == "calling echo"
    assert sent[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"message": "hi"})}}
    ]
    # tool result -> tool role message
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "hi"}


def test_chat_forwards_api_key_and_api_base(fake_completion):
    fake_completion.return_value = _make_text_response("ok")
    client = LiteLLMClient(
        model="openai/openhands/glm-5.2",
        api_key="sk-test",
        api_base="https://llm-proxy.app.all-hands.dev",
    )
    client.chat(messages=[{"role": "user", "content": "hi"}])
    sent = fake_completion.calls[0]
    assert sent["api_key"] == "sk-test"
    assert sent["api_base"] == "https://llm-proxy.app.all-hands.dev"
    assert sent["model"] == "openai/openhands/glm-5.2"


def test_chat_omits_api_key_and_api_base_when_unset(fake_completion):
    fake_completion.return_value = _make_text_response("ok")
    client = _new_client()
    client.chat(messages=[{"role": "user", "content": "hi"}])
    sent = fake_completion.calls[0]
    assert "api_key" not in sent
    assert "api_base" not in sent
    assert "tools" not in sent


def test_chat_passes_max_tokens_and_extra_kwargs(fake_completion):
    fake_completion.return_value = _make_text_response("ok")
    client = _new_client(max_tokens=123)
    client.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=999)
    sent = fake_completion.calls[0]
    assert sent["max_tokens"] == 999  # explicit override wins
    assert sent["temperature"] == 0.2


def test_chat_stream_aggregates_text_and_emits_done(fake_completion):
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hel", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
        ),
    ]
    fake_completion.return_value = chunks
    client = _new_client()
    events = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    text_events = [e for e in events if e.type == "text"]
    assert [e.delta for e in text_events] == ["hel", "lo"]
    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    final = done_events[0].delta
    assert final.content == "hello"
    assert final.stop_reason == "end_turn"
    assert final.usage == {"input_tokens": 2, "output_tokens": 2}
    # streaming request must set stream=True
    assert fake_completion.calls[0].get("stream") is True


def test_chat_stream_collects_tool_calls(fake_completion):
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="c1",
                                type="function",
                                function=SimpleNamespace(name="echo", arguments=json.dumps({"message": "x"})),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="tool_calls")],
            usage=None,
        ),
    ]
    fake_completion.return_value = chunks
    client = _new_client()
    events = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    final = [e for e in events if e.type == "done"][0].delta
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "echo"
    assert final.tool_calls[0].arguments == {"message": "x"}
    assert final.stop_reason == "tool_use"


def test_map_stop_reason_unknown_passthrough():
    assert LiteLLMClient._map_stop_reason("length") == "max_tokens"
    assert LiteLLMClient._map_stop_reason("weird-finish") == "weird-finish"
    assert LiteLLMClient._map_stop_reason(None) is None


def test_invalid_tool_arguments_falls_back_to_raw(fake_completion):
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            type="function",
                            function=SimpleNamespace(name="echo", arguments="not-json{"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )
    fake_completion.return_value = raw
    client = _new_client()
    resp = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.tool_calls[0].arguments == {"_raw": "not-json{"}
