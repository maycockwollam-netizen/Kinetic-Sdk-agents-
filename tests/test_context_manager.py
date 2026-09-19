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
    ContextBudget,
    ContextManager,
    LLMContextSummarizer,
    NoopContextManager,
    SimpleTruncateContextManager,
    SummarizingContextManager,
    ToolOutputCompressor,
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
    """user request, then ``turns`` rounds of assistant + tool_result.

    The assistant turns carry real ``tool_use`` blocks — the exact shape the
    agent loop stores — so every tool_result has a matching tool_use.
    """
    state = ConversationState(system_prompt="You are a coding agent.")
    state.messages.append(_user("Fix the flaky test in the payments module."))
    for i in range(turns):
        state.messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"step {i}: " + "x" * filler},
                    {"type": "tool_use", "id": f"call_{i}", "name": "t", "input": {}},
                ],
            }
        )
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


def test_section_budget_prioritizes_tool_output_before_history():
    state = ConversationState()
    state.messages = [_user("history " * 20), _tool_result("call", "log " * 300)]
    manager = SimpleTruncateContextManager(
        budget=ContextBudget(tool_output=0.10, history=0.80, system_prompt=0.0, memory=0.0, plan=0.0),
        tool_output_compressor=ToolOutputCompressor(),
    )

    assert manager.should_compact(state, model_context_limit=500)
    compacted = manager.compact(state)

    assert compacted.messages[0] == state.messages[0]
    block = compacted.messages[1]["content"][0]
    assert block["_compaction"]["kind"] == "truncated"
    assert manager.budget_report(state)["tool_output"] > manager.budget_report(state)["history"]


def test_structure_aware_compressor_keeps_path_and_exit_code():
    path_line = "/workspace/project/src/main.py:42: Error: failed assertion\n"
    exit_line = "process exited with 17\n"
    text = "noise\n" * 50 + path_line + "noise\n" * 50 + exit_line + "noise\n" * 50
    compressed = ToolOutputCompressor().compress(text, 180)

    assert path_line.strip() in compressed
    assert exit_line.strip() in compressed


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


# --- parallel tool-call group integrity (regression: orphaned tool_result) --


def _parallel_group(prefix: str, n: int, output_filler: int = 60) -> list[dict]:
    """ONE assistant message with N tool_use blocks + N tool_result messages.

    This is the exact shape the agent loop stores when the model issues
    parallel tool calls in a single turn.
    """
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"{prefix}{i}", "name": "t", "input": {}}
                for i in range(n)
            ],
        }
    ]
    msgs.extend(
        _tool_result(f"{prefix}{i}", f"out-{prefix}{i} " + "y" * output_filler)
        for i in range(n)
    )
    return msgs


def _assert_tool_block_integrity(messages: list[dict]) -> None:
    """Every tool_result must follow its tool_use; every tool_use must have
    its result (providers reject histories violating either)."""
    seen_uses: set[str] = set()
    open_uses: set[str] = set()
    for msg in messages:
        content = msg["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                seen_uses.add(block["id"])
                open_uses.add(block["id"])
            elif block.get("type") == "tool_result":
                assert block["tool_use_id"] in seen_uses, (
                    f"orphaned tool_result {block['tool_use_id']}"
                )
                open_uses.discard(block["tool_use_id"])
    assert not open_uses, f"dangling tool_use without result: {open_uses}"


def test_compact_never_orphans_parallel_tool_results():
    """Regression: the old anchor could land mid-parallel-group, keeping
    tool_results whose tool_use was elided -> provider rejects the next call."""
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("task gốc"))
    for prefix, n in (("A", 2), ("B", 3), ("C", 2), ("D", 2)):
        state.messages.extend(_parallel_group(prefix, n))

    manager = SimpleTruncateContextManager(keep_last_tool_results=5)
    compacted = manager.compact(state)

    _assert_tool_block_integrity(compacted.messages)
    # Compaction still happened (a placeholder replaced the middle span).
    assert any(
        isinstance(m["content"], str) and "rút gọn" in m["content"]
        for m in compacted.messages
    )
    # The boundary widened back past the whole B group: either all of B's
    # results survive together with their tool_use, or none do.
    b_results = [
        b["tool_use_id"]
        for m in compacted.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    b_survivors = [i for i in b_results if i.startswith("B")]
    assert b_survivors in ([], ["B0", "B1", "B2"])


def test_compact_parallel_groups_integrity_fuzz():
    """Integrity holds across many group-size / keep-count combinations."""
    for keep in range(1, 7):
        for sizes in ((1, 1, 1, 1), (2, 3, 1, 4), (5,), (1, 2)):
            state = ConversationState(system_prompt="sys")
            state.messages.append(_user("task"))
            for gi, n in enumerate(sizes):
                state.messages.extend(_parallel_group(f"g{gi}_", n, output_filler=5))
            manager = SimpleTruncateContextManager(keep_last_tool_results=keep)
            compacted = manager.compact(state)
            _assert_tool_block_integrity(compacted.messages)


def test_compact_head_with_tool_blocks_is_not_kept_alone():
    """A first message carrying tool blocks must not survive in isolation."""
    state = ConversationState()
    state.messages.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "x0", "name": "t", "input": {}}],
        }
    )
    state.messages.append(_tool_result("x0", "done " + "y" * 100))
    state.messages.extend(_parallel_group("B", 2, output_filler=100))
    state.messages.extend(_parallel_group("C", 2, output_filler=100))

    manager = SimpleTruncateContextManager(keep_last_tool_results=2)
    compacted = manager.compact(state)

    _assert_tool_block_integrity(compacted.messages)
    assert compacted.messages[0] is not state.messages[0]


def test_compact_unresolvable_tool_reference_returns_unchanged():
    """A corrupted tail (result without any use anywhere) must degrade to no
    compaction instead of emitting another invalid history."""
    state = ConversationState()
    state.messages.append(_user("task"))
    state.messages.extend(_parallel_group("A", 2))
    state.messages.extend(_parallel_group("B", 2))
    # Ghost sits INSIDE the protected tail: no cut can ever pair it.
    state.messages.append(_tool_result("ghost", "no matching tool_use anywhere"))

    manager = SimpleTruncateContextManager(keep_last_tool_results=3)
    compacted = manager.compact(state)

    assert len(compacted.messages) == len(state.messages)
    assert compacted.messages == state.messages
    assert compacted is not state


def test_compact_elides_orphaned_result_left_behind_the_tail():
    """An orphaned result that falls into the elided span is dropped, which
    actually REPAIRS integrity for the surviving tail."""
    state = ConversationState()
    state.messages.append(_user("task"))
    state.messages.append(_tool_result("ghost", "orphan from an older bug"))
    state.messages.extend(_parallel_group("A", 2))
    state.messages.extend(_parallel_group("B", 2))

    manager = SimpleTruncateContextManager(keep_last_tool_results=3)
    compacted = manager.compact(state)

    _assert_tool_block_integrity(compacted.messages)
    assert all(
        not (
            isinstance(m["content"], list)
            and any(
                isinstance(b, dict) and b.get("tool_use_id") == "ghost"
                for b in m["content"]
            )
        )
        for m in compacted.messages
    )


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
    elided_text = state.messages[1]["content"][0]["text"]
    assert elided_text not in summary


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


# --- SummarizingContextManager: prompt, fallback event, redaction, parity -----


def test_summarizing_manager_accepts_summarizer_client_directly():
    """An LLMClient alone is enough; it is wrapped in LLMContextSummarizer."""
    llm = MockLLM([text_response("Tóm tắt từ client.")])
    manager = SummarizingContextManager(keep_last_tool_results=2, summarizer_client=llm)

    compacted = manager.compact(_long_conversation(turns=5))

    assert isinstance(manager.summarizer, LLMContextSummarizer)
    assert "Tóm tắt từ client." in compacted.messages[1]["content"]
    # The summary call uses the low max_tokens cap (the LLM gets no tools).
    assert llm.calls[0]["kwargs"] == {"max_tokens": 150}
    assert llm.calls[0]["tools"] is None


def test_summarizing_manager_rejects_both_summarizer_and_client():
    with pytest.raises(ValueError, match="not both"):
        SummarizingContextManager(
            summarizer=FakeSummarizer(), summarizer_client=MockLLM([])
        )


def test_summarizing_manager_prompt_contains_elided_content():
    """The summarizer receives exactly the elided middle span."""
    summarizer = FakeSummarizer()
    manager = SummarizingContextManager(keep_last_tool_results=2, summarizer=summarizer)
    state = _long_conversation(turns=5)

    manager.compact(state)

    assert len(summarizer.calls) == 1
    elided = summarizer.calls[0]
    contents = [str(m["content"]) for m in elided]
    # The span is the middle: neither the first user message nor the tail.
    assert state.messages[0]["content"] not in contents
    assert any("step 0" in c for c in contents)
    # The tail (last 2 tool results + neighbours) is NOT sent for summarization.
    assert not any("call_4" in c for c in contents)


def test_summarizing_manager_failure_emits_event_and_falls_back():
    bus = EventBus()
    events = []
    bus.subscribe("context.summarization_failed", events.append)
    manager = SummarizingContextManager(
        keep_last_tool_results=2, summarizer=RaisingSummarizer(), event_bus=bus
    )

    compacted = manager.compact(_long_conversation(turns=5))  # must not raise

    # Fallback placeholder replaces the middle, exactly like truncation.
    assert any("rút gọn" in str(m["content"]) for m in compacted.messages)
    assert len(events) == 1
    payload = events[0].payload
    assert events[0].type == "context.summarization_failed"
    assert payload["reason"] == "exception"
    assert payload["elided_messages"] > 0
    assert "summarizer unavailable" in payload["error"]
    assert payload["manager"] == "SummarizingContextManager"


def test_summarizing_manager_failure_event_without_bus_does_not_crash():
    manager = SummarizingContextManager(
        keep_last_tool_results=2, summarizer=RaisingSummarizer()
    )
    compacted = manager.compact(_long_conversation(turns=5))
    assert any("rút gọn" in str(m["content"]) for m in compacted.messages)


def test_summarizing_manager_empty_and_nonstring_summary_emit_event():
    for summarizer, reason in (
        (FakeSummarizer(summary=""), "empty_summary"),
        (NoneSummarizer(), "non_string_summary"),
    ):
        bus = EventBus()
        events = []
        bus.subscribe("context.summarization_failed", events.append)
        manager = SummarizingContextManager(
            keep_last_tool_results=2, summarizer=summarizer, event_bus=bus
        )
        compacted = manager.compact(_long_conversation(turns=5))
        assert any("rút gọn" in str(m["content"]) for m in compacted.messages)
        assert [e.payload["reason"] for e in events] == [reason]


def test_summarizing_manager_redacts_secrets_before_summarizing():
    secret = "ghp_" + "a1B2c3" * 6  # matches the GitHub token pattern
    summarizer = FakeSummarizer()
    manager = SummarizingContextManager(keep_last_tool_results=2, summarizer=summarizer)
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("task"))
    for i in range(4):
        state.messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"step {i} " + "x" * 100},
                    {"type": "tool_use", "id": f"c{i}", "name": "t", "input": {}},
                ],
            }
        )
        output = f"api_key = {secret}" if i == 0 else "ok " + "y" * 100
        state.messages.append(_tool_result(f"c{i}", output))

    manager.compact(state)

    assert summarizer.calls
    sent = str(summarizer.calls[0])
    assert secret not in sent
    assert "[REDACTED]" in sent


def test_summarizing_and_simple_manager_agree_on_kept_structure():
    """Same input: identical head/tail, only the elided replacement differs."""
    simple = SimpleTruncateContextManager(keep_last_tool_results=2)
    summarizing = SummarizingContextManager(
        keep_last_tool_results=2, summarizer=FakeSummarizer()
    )
    state = _long_conversation(turns=6)

    a = simple.compact(state)
    b = summarizing.compact(state)

    assert len(a.messages) == len(b.messages)
    # Head (first user message) and protected tail are identical.
    assert a.messages[0] == b.messages[0] == state.messages[0]
    assert a.messages[2:] == b.messages[2:]
    assert a.system_prompt == b.system_prompt == state.system_prompt
    # Only the replacement for the elided span differs.
    assert "rút gọn" in a.messages[1]["content"]
    assert "tóm tắt" in b.messages[1]["content"]


def test_agent_wires_bus_into_summarizing_manager():
    """The agent's bus receives context.summarization_failed during run()."""
    bus = EventBus()
    events = []
    bus.subscribe("context.summarization_failed", events.append)
    manager = SummarizingContextManager(
        keep_last_tool_results=2, summarizer=RaisingSummarizer()
    )
    agent = Agent(
        llm=MockLLM([text_response("done")]),
        tools=[EchoTool()],
        state=_long_conversation(turns=8, filler=400),
        event_bus=bus,
        context_manager=manager,
        model_context_limit=500,
    )

    assert agent.run() == "done"
    assert len(events) == 1
    # The event flows through the agent's bus, sharing its stream (the
    # manager publishes directly, so it carries no run_id of its own).
    assert events[0].payload["reason"] == "exception"


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



# --- pluggable token counters --------------------------------------------------


def test_custom_token_counter_replaces_heuristic():
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("x" * 100))

    default = SimpleTruncateContextManager()
    counting_chars = SimpleTruncateContextManager(token_counter=len)

    assert default.estimate_state_tokens(state) < counting_chars.estimate_state_tokens(state)
    # The counter is consulted for every chunk (system + message).
    seen: list[int] = []
    SimpleTruncateContextManager(
        token_counter=lambda t: seen.append(len(t)) or len(t)
    ).estimate_state_tokens(state)
    assert len(seen) == 2


def test_chars_per_token_constructor_arg_is_respected():
    """Regression: chars_per_token used to be stored but never consulted."""
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("x" * 400))

    coarse = SimpleTruncateContextManager(chars_per_token=4)
    fine = SimpleTruncateContextManager(chars_per_token=1)

    assert fine.estimate_state_tokens(state) > coarse.estimate_state_tokens(state)
    # fine fires compaction at limits coarse still considers safe.
    limit = coarse.estimate_state_tokens(state)
    assert coarse.should_compact(state, model_context_limit=int(limit / 0.8) + 10) is False
    assert fine.should_compact(state, model_context_limit=int(limit / 0.8) + 10) is True


def test_tiktoken_counter_counts_real_tokens():
    pytest.importorskip("tiktoken")
    from kinetic_sdk.context.tokens import TiktokenCounter

    counter = TiktokenCounter()
    vietnamese = "Xin chào, đây là một câu tiếng Việt đầy đủ dấu."
    real = counter(vietnamese)
    heuristic = estimate_tokens(vietnamese)
    # cl100k splits Vietnamese into many more tokens than len//4 suggests.
    assert real > heuristic


def test_tiktoken_counter_integrates_with_manager():
    pytest.importorskip("tiktoken")
    from kinetic_sdk.context.tokens import TiktokenCounter

    manager = SimpleTruncateContextManager(token_counter=TiktokenCounter())
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("một đoạn văn bản tiếng Việt " * 20))
    assert manager.estimate_state_tokens(state) > 0


def test_tiktoken_counter_never_returns_zero_or_negative():
    pytest.importorskip("tiktoken")
    from kinetic_sdk.context.tokens import TiktokenCounter

    counter = TiktokenCounter()
    assert counter("") >= 1


# --- oversized tool-result content truncation ---------------------------------


def _big_result_conversation(big_chars: int = 20_000, turns: int = 4) -> ConversationState:
    state = ConversationState(system_prompt="sys")
    state.messages.append(_user("task"))
    for i in range(turns):
        state.messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"b{i}", "name": "t", "input": {}}
                ],
            }
        )
        size = big_chars if i == turns - 1 else 200
        state.messages.append(_tool_result(f"b{i}", "z" * size))
    return state


def test_compact_truncates_oversized_tool_result_in_tail():
    manager = SimpleTruncateContextManager(
        keep_last_tool_results=2, max_tool_result_chars=1_000
    )
    compacted = manager.compact(_big_result_conversation())

    results = [
        b
        for m in compacted.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    biggest = max(results, key=lambda b: len(b["content"]))
    assert "cắt bớt" in biggest["content"]
    # head + marker + tail stays close to the limit (marker overhead aside).
    assert len(biggest["content"]) < 1_200
    _assert_tool_block_integrity(compacted.messages)


def test_compact_truncation_applies_when_no_message_can_be_elided():
    """The protected-tail-overflow case: everything is kept, but giant tool
    results are still cut - the case elision alone could never fix."""
    manager = SimpleTruncateContextManager(
        keep_last_tool_results=10, max_tool_result_chars=1_000
    )
    state = _big_result_conversation(turns=2)
    compacted = manager.compact(state)

    assert len(compacted.messages) == len(state.messages)  # nothing elided
    last = compacted.messages[-1]["content"][0]["content"]
    assert "cắt bớt" in last


def test_truncation_preserves_original_state():
    manager = SimpleTruncateContextManager(
        keep_last_tool_results=2, max_tool_result_chars=1_000
    )
    state = _big_result_conversation()
    original_last = state.messages[-1]["content"][0]["content"]

    manager.compact(state)

    assert state.messages[-1]["content"][0]["content"] == original_last
    assert "cắt bớt" not in original_last


def test_truncation_leaves_small_results_untouched():
    manager = SimpleTruncateContextManager(
        keep_last_tool_results=2, max_tool_result_chars=1_000
    )
    state = _big_result_conversation(big_chars=200)
    compacted = manager.compact(state)

    assert all(
        "cắt bớt" not in str(m["content"]) for m in compacted.messages
    )


def test_truncation_disabled_by_default():
    manager = SimpleTruncateContextManager(keep_last_tool_results=2)
    compacted = manager.compact(_big_result_conversation())
    results = [
        b
        for m in compacted.messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert max(len(b["content"]) for b in results) == 20_000


def test_max_tool_result_chars_validation():
    with pytest.raises(ValueError):
        SimpleTruncateContextManager(max_tool_result_chars=10)


def test_summarizing_manager_also_truncates():
    manager = SummarizingContextManager(
        keep_last_tool_results=2,
        summarizer=FakeSummarizer(),
        max_tool_result_chars=1_000,
    )
    compacted = manager.compact(_big_result_conversation())
    assert any("cắt bớt" in str(m["content"]) for m in compacted.messages)
    assert any("tóm tắt" in str(m["content"]) for m in compacted.messages)


def test_elided_message_has_truncation_provenance_metadata():
    manager = SimpleTruncateContextManager(keep_last_tool_results=1)
    compacted = manager.compact(_long_conversation(turns=4))

    marker = next(message for message in compacted.messages if message.get("_compaction"))
    assert marker["_compaction"] == {"kind": "truncated", "source_message_count": 6}

# --- advanced optional compaction ------------------------------------------


def test_provider_token_counter_uses_provider_result_and_caches():
    from kinetic_sdk.context.tokens import ProviderTokenCounter

    class CountingLLM(MockLLM):
        def __init__(self):
            super().__init__([])
            self.count_calls = 0

        def count_tokens(self, text: str) -> int:
            self.count_calls += 1
            return 37

    llm = CountingLLM()
    counter = ProviderTokenCounter(llm, fallback=lambda _: 1)

    assert counter("same text") == 37
    assert counter("same text") == 37
    assert llm.count_calls == 1


def test_provider_token_counter_falls_back_when_provider_counting_fails():
    from kinetic_sdk.context.tokens import ProviderTokenCounter

    class FailingLLM(MockLLM):
        def count_tokens(self, text: str) -> int:
            raise RuntimeError("provider rate limited")

    assert ProviderTokenCounter(FailingLLM([]), fallback=lambda text: len(text))("abcd") == 4


def test_tail_token_budget_selects_recent_results_by_cost_and_respects_cap():
    def conversation(result_sizes: list[int]) -> ConversationState:
        state = ConversationState()
        state.messages.append(_user("task"))
        for index, size in enumerate(result_sizes):
            state.messages.append(
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": f"tail-{index}", "name": "t", "input": {}},
                ]}
            )
            state.messages.append(_tool_result(f"tail-{index}", "x" * size))
        return state

    # With character counting, the newest short result fits but the preceding
    # long result exhausts the 300-character tail allowance.
    constrained = SimpleTruncateContextManager(
        keep_last_tool_results=3, token_counter=len, tail_token_budget=0.15
    )
    constrained_state = conversation([20, 20, 500, 20])
    constrained.should_compact(constrained_state, model_context_limit=2_000)
    constrained_result = constrained.compact(constrained_state)
    constrained_ids = [
        message["content"][0]["tool_use_id"] for message in constrained_result.messages
        if isinstance(message["content"], list) and message["content"] and message["content"][0].get("type") == "tool_result"
    ]
    assert constrained_ids == ["tail-3"]

    # Small results use the same budget more efficiently, but never exceed the
    # original count-based maximum.
    roomy = SimpleTruncateContextManager(
        keep_last_tool_results=3, token_counter=len, tail_token_budget=0.15
    )
    roomy_state = conversation([20, 20, 20, 20])
    roomy.should_compact(roomy_state, model_context_limit=2_000)
    roomy_result = roomy.compact(roomy_state)
    roomy_ids = [
        message["content"][0]["tool_use_id"] for message in roomy_result.messages
        if isinstance(message["content"], list) and message["content"] and message["content"][0].get("type") == "tool_result"
    ]
    assert roomy_ids == ["tail-1", "tail-2", "tail-3"]
    assert len(roomy_ids) <= 3


def test_progressive_summarizer_merges_structured_summary_across_compactions():
    from kinetic_sdk.context.manager import ProgressiveSummarizer, StructuredSummary

    first = '{"decisions":["use cache"],"files_touched":["a.py"],"errors_encountered":[],"open_questions":["ship?"]}'
    second = '{"decisions":["add tests"],"files_touched":["b.py"],"errors_encountered":["timeout"],"open_questions":[]}'
    progressive = ProgressiveSummarizer(MockLLM([text_response(first), text_response(second)]))
    manager = SummarizingContextManager(keep_last_tool_results=1, summarizer=progressive)

    once = manager.compact(_long_conversation(turns=4))
    twice_source = ConversationState(
        system_prompt=once.system_prompt,
        messages=[*once.messages, *_long_conversation(turns=3).messages[1:]],
        metadata=dict(once.metadata),
    )
    twice = manager.compact(twice_source)

    summary = twice.metadata["structured_summary"]
    assert isinstance(summary, StructuredSummary)
    assert summary.decisions == ["use cache", "add tests"]
    assert summary.files_touched == ["a.py", "b.py"]
    assert summary.errors_encountered == ["timeout"]
    assert summary.open_questions == ["ship?"]
