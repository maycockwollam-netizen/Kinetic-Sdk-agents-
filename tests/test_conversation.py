"""Unit tests for ConversationState."""

from __future__ import annotations

from kinetic_sdk.conversation.state import ConversationState


def test_empty_state_for_llm():
    state = ConversationState(system_prompt="sys")
    system, messages = state.for_llm()
    assert system == "sys"
    assert messages == []


def test_add_user_message():
    state = ConversationState()
    state.add_user_message("hello")
    assert state.length == 1
    assert state.messages[0] == {"role": "user", "content": "hello"}


def test_add_assistant_text():
    state = ConversationState()
    state.add_assistant_text("hi")
    assert state.messages[0] == {"role": "assistant", "content": "hi"}


def test_add_tool_result_uses_anthropic_shape():
    state = ConversationState()
    state.add_tool_result("call_1", "output text")
    msg = state.messages[0]
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "output text"
    assert "is_error" not in block


def test_add_tool_result_with_error_flag():
    state = ConversationState()
    state.add_tool_result("c1", "boom", is_error=True)
    block = state.messages[0]["content"][0]
    assert block["is_error"] is True


def test_add_tool_result_serialises_non_string_output():
    state = ConversationState()
    state.add_tool_result("c1", {"value": 42})
    block = state.messages[0]["content"][0]
    assert block["content"] == "{'value': 42}"


def test_max_messages_cap_drops_oldest():
    state = ConversationState(max_messages=3)
    for i in range(5):
        state.add_user_message(f"m{i}")
    # Cap is enforced but keeps at least one message.
    assert state.length <= 3
    assert state.length >= 1


def test_reset_keeps_system_prompt():
    state = ConversationState(system_prompt="sys")
    state.add_user_message("hi")
    state.reset()
    assert state.length == 0
    assert state.system_prompt == "sys"


def test_for_llm_returns_copy():
    state = ConversationState()
    state.add_user_message("hi")
    _, msgs = state.for_llm()
    msgs.append({"role": "user", "content": "injected"})
    assert state.length == 1
