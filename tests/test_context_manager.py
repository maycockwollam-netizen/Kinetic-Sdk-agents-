"""Tests for the Stage 2 context manager + its wiring into the agent loop.

Covers the token heuristic, should_compact thresholds, the truncation
compaction policy (preserve head + recent tool results, immutable-style),
edge cases on very short conversations, and the agent-loop integration
(``context.compacted`` event + the request actually sent to the LLM).
"""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.context.manager import (
    ContextManager,
    LLMContextSummarizer,
    NoopContextManager,
    SimpleTruncateContextManager,
    SummarizingContextManager,
    estimate_tokens,
)
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import EventBus
from tests._helpers import EchoTool, MockLLM, text_response


class FakeSummarizer:
    def __init__(self, summary: str = "Đã phân tích lỗi flaky test và giữ lại log quan trọng.") -> None:
        self.summary = summary
        self.calls: list[list[dict]] = []

    def summarize(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.summary


class RaisingSummarizer:
    def summarize(self, messages: list[dict]) -> str:
        raise RuntimeError("summarizer unavailable")


class NoneSummarizer:
    def summarize(self, messages: list[dict]) -> None:
        return None


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool_result(call_id: str, output: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}],
    }


def _long_conversation(turns: int = 8, filler: int = 200) -> ConversationState:
    """user request, then ``turns`` rounds of assistant + tool_result."""
    state = ConversationState(system_prompt="You are a coding agent.")
    state.messages.append(_user("Fix the flaky test in the payments module."))
    for i in range(turns):
        state.messages.append(_assistant(f"step {i}: " + "x" * filler))
        state.messages.append(_tool_result(f"call_{i}", "result " + "y" * filler))
    return state


# --- token estimation ------------------------------------------------------


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
    # Empty-ish input still reports a floor of 1 token.
    assert estimate_tokens("") == 1


# --- should_compact --------------------------------------------------------


def test_should_compact_false_for_short_conversation():
    manager = SimpleTruncateContextManager()
    state = ConversationState(system_prompt="hi")
    state.messages.append(_user("short question"))
    assert manager.should_compact(state, model_context_limit=128_000) is False


def test_should_compact_true_past_threshold():
    manager = SimpleTruncateContextManager(safety_threshold=0.8)
    state = ConversationState()
    state.messages.append(_user("x" * 400))  # ~100 tokens
    # 0.8 * 100 = 80 < 100 -> over budget.
    assert manager.should_compact(state, model_context_limit=100) is True
    # 0.8 * 1000 = 800 > 100 -> comfortably under budget.
    assert manager.should_compact(state, model_context_limit=1000) is False


def test_should_compact_rejects_nonpositive_limit():
    manager = SimpleTruncateContextManager()
    with pytest.raises(ValueError):
        manager.should_compact(ConversationState(), model_context_limit=0)


# --- compact policy --------------------------------------------------------


def test_compact_preserves_system_prompt_first_user_message_and_recent_tool_results():
    manager = SimpleTruncateContextManager(keep_last_tool_results=3)
    state = _long_conversation(turns=8)
    compacted = manager.compact(state)

    assert compacted.system_prompt == state.system_prompt
    assert compacted.messages[0] == state.messages[0]

    compacted_tool_results = [
        m for m in compacted.messages if any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in (m["content"] if isinstance(m["content"], list) else [])
        )
    ]
    # Exactly the 3 most recent tool results survive, in order.
    assert [m["content"][0]["tool_use_id"] for m in compacted_tool_results] == [
        "call_5",
        "call_6",
        "call_7",
    ]


def test_compact_inserts_placeholder_with_removed_count():
    manager = SimpleTruncateContextManager(keep_last_tool_results=3)
    state = _long_conversation(turns=8)
    original_len = len(state.messages)
    compacted = manager.compact(state)

    removed = original_len - (len(compacted.messages) - 1)  # -1 for the placeholder itself
    placeholder = compacted.messages[1]
    assert placeholder["role"] == "user"
    assert placeholder["content"] == f"[{removed} tin nhắn trước đó đã được rút gọn]"
    assert removed > 0
    assert len(compacted.messages) < original_len


def test_compact_does_not_mutate_original_state():
    manager = SimpleTruncateContextManager(keep_last_tool_results=2)
    state = _long_conversation(turns=6)
    snapshot = [dict(m) for m in state.messages]

    compacted = manager.compact(state)

    assert state.messages == snapshot  # reference holder sees no change
    assert compacted is not state
    assert compacted.messages is not state.messages


def test_compact_short_conversations_are_untouched():
    manager = SimpleTruncateContextManager()
    for n in (0, 1, 2):
        state = ConversationState(system_prompt="sys")
        for i in range(n):
            state.messages.append(_user(f"msg {i}"))
        compacted = manager.compact(state)
        assert [dict(m) for m in compacted.messages] == [dict(m) for m in state.messages]


def test_compact_without_any_tool_results_keeps_head_and_last_turn():
    manager = SimpleTruncateContextManager()
    state = ConversationState()
    for i in range(6):
        state.messages.append(_user(f"msg {i}"))
    compacted = manager.compact(state)
    assert compacted.messages[0] == state.messages[0]
    assert compacted.messages[-1] == state.messages[-1]
    assert "rút gọn" in compacted.messages[1]["content"]


def test_compact_when_everything_is_protected_returns_copy():
    # Only 2 tool results but keep_last_tool_results=5 -> tail covers
    # (almost) everything, so nothing is removed.
    manager = SimpleTruncateContextManager(keep_last_tool_results=5)
    state = ConversationState()
    state.messages.append(_user("task"))
    state.messages.append(_assistant("working"))
    state.messages.append(_tool_result("c1", "ok"))
    compacted = manager.compact(state)
    assert len(compacted.messages) == len(state.messages)
    assert compacted is not state


def test_noop_manager_never_compacts_and_copies():
    manager = NoopContextManager()
    state = _long_conversation(turns=8)
    assert manager.should_compact(state, model_context_limit=1) is False
    copied = manager.compact(state)
    assert copied.messages == state.messages
    assert copied is not state


def test_summarizing_manager_uses_summary_for_elided_span():
    assert issubclass(SummarizingContextManager, SimpleTruncateContextManager)
    summarizer = FakeSummarizer()
    manager = SummarizingContextManager(keep_last_tool_results=2, summarizer=summarizer)
    state = _long_conversation(turns=5)

    compacted = manager.compact(state)

    assert summarizer.calls
    assert compacted.messages[0] == state.messages[0]
    summary = compacted.messages[1]["content"]
    assert "đã được tóm tắt" in summary
    assert summarizer.summary in summary
    assert "rút gọn" not in summary
    assert state.messages[1]["content"] not in summary


def test_summarizing_manager_falls_back_to_truncation_without_summarizer():
    manager = SummarizingContextManager(keep_last_tool_results=2)
    compacted = manager.compact(_long_conversation(turns=5))
    assert any("rút gọn" in str(m["content"]) for m in compacted.messages)


def test_summarizing_manager_falls_back_to_truncation_on_failure_or_empty_summary():
    for summarizer in (RaisingSummarizer(), FakeSummarizer(summary=""), NoneSummarizer()):
        manager = SummarizingContextManager(keep_last_tool_results=2, summarizer=summarizer)
        compacted = manager.compact(_long_conversation(turns=5))
        assert any("rút gọn" in str(m["content"]) for m in compacted.messages)
        assert "None" not in str(compacted.messages[1]["content"])


def test_summarizing_manager_truncates_long_summary():
    manager = SummarizingContextManager(
        keep_last_tool_results=2,
        summarizer=FakeSummarizer(summary="x" * 50),
        max_summary_chars=10,
    )
    compacted = manager.compact(_long_conversation(turns=5))
    summary = compacted.messages[1]["content"]
    assert "xxxxxxxxxx…" in summary


def test_llm_context_summarizer_uses_injected_llm():
    llm = MockLLM([text_response("Tóm tắt ngắn.")])
    summarizer = LLMContextSummarizer(llm, max_tokens=42)

    assert summarizer.summarize([_user("xin chào")]) == "Tóm tắt ngắn."
    call = llm.calls[0]
    assert call["kwargs"] == {"max_tokens": 42}
    # The system prompt goes through the dedicated `system` parameter, per the
    # LLMClient contract, not as a message in the history.
    assert "Tóm tắt" in call["system"]
    assert all(m["role"] != "system" for m in call["messages"])
    assert "xin chào" in call["messages"][0]["content"]


def test_simple_manager_is_a_context_manager():
    assert isinstance(SimpleTruncateContextManager(), ContextManager)


# --- agent loop integration -------------------------------------------------


def test_agent_compacts_before_llm_call_and_emits_event():
    manager = SimpleTruncateContextManager(keep_last_tool_results=2)
    state = _long_conversation(turns=8, filler=400)
    original_messages = len(state.messages)
    bus = EventBus()
    events = []
    bus.subscribe("context.compacted", events.append)

    llm = MockLLM([text_response("done")])
    agent = Agent(
        llm=llm,
        tools=[EchoTool()],
        state=state,
        event_bus=bus,
        context_manager=manager,
        model_context_limit=500,  # far below the conversation's estimate
    )
    result = agent.run()

    assert result == "done"
    assert len(events) == 1
    payload = events[0].payload
    assert payload["manager"] == "SimpleTruncateContextManager"
    assert payload["messages_before"] == original_messages
    assert payload["messages_after"] < payload["messages_before"]
    assert payload["messages_removed"] == original_messages - payload["messages_after"]

    # The request actually sent to the LLM is the compacted history.
    sent = llm.calls[0]["messages"]
    assert len(sent) == payload["messages_after"]
    assert manager.estimate_state_tokens(agent.state) < 500 * 4  # well under the fake limit
    # Original system prompt survives compaction.
    assert llm.calls[0]["system"] == "You are a coding agent."


def test_agent_does_not_compact_when_under_threshold():
    bus = EventBus()
    events = []
    bus.subscribe("context.compacted", events.append)
    llm = MockLLM([text_response("hi")])
    agent = Agent(llm=llm, event_bus=bus, model_context_limit=128_000)
    assert agent.run("hello") == "hi"
    assert events == []


def test_agent_can_disable_compaction_with_noop_manager():
    bus = EventBus()
    events = []
    bus.subscribe("context.compacted", events.append)
    state = _long_conversation(turns=8, filler=400)
    original_len = len(state.messages)
    llm = MockLLM([text_response("done")])
    agent = Agent(
        llm=llm,
        state=state,
        event_bus=bus,
        context_manager=NoopContextManager(),
        model_context_limit=1,
    )
    agent.run()
    assert events == []
    assert len(llm.calls[0]["messages"]) == original_len


def test_agent_default_context_manager_is_simple_truncate():
    agent = Agent(llm=MockLLM([text_response("x")]))
    assert isinstance(agent.context_manager, SimpleTruncateContextManager)
    assert agent.model_context_limit == Agent.DEFAULT_MODEL_CONTEXT_LIMIT
