"""Regression tests for bounded progressive structured summaries."""

from __future__ import annotations

import json

from kinetic_sdk.context.manager import (
    ProgressiveSummarizer,
    StructuredSummary,
    SummarizingContextManager,
)
from kinetic_sdk.context.tokens import ProviderTokenCounter
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.testing import MockLLMClient, text_response


def _summary(*, decision: str = "", file_name: str = "", error: str = "", question: str = "", goal: str = "") -> str:
    return json.dumps({
        "goal": [goal] if goal else [], "decisions": [decision] if decision else [],
        "files_touched": [file_name] if file_name else [],
        "errors_encountered": [error] if error else [], "open_questions": [question] if question else [],
    }, ensure_ascii=False)


def _state() -> ConversationState:
    return ConversationState(messages=[
        {"role": "user", "content": "mục tiêu ban đầu"},
        *({"role": "user", "content": f"turn {i}"} for i in range(4)),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
    ])


def test_progressive_prompt_keeps_recent_errors_questions_and_is_not_tail_cut():
    manager = SummarizingContextManager(
        keep_last_tool_results=1, summarizer=ProgressiveSummarizer(MockLLMClient([])),
        max_structured_chars=4_000,
    )
    result = StructuredSummary()
    for i in range(15):
        result = result.merged(
            StructuredSummary(decisions=[f"d{i}"], files_touched=[f"f{i}"])
        )
    result = result.merged(StructuredSummary(errors_encountered=["LỖI mới nhất"], open_questions=["CÂU HỎI MỞ mới nhất"]))
    bounded = result.limited(manager.max_structured_chars, manager.max_items_per_field)
    prompt = bounded.to_prompt()
    assert "LỖI mới nhất" in prompt and "CÂU HỎI MỞ mới nhất" in prompt
    assert len(prompt) <= manager.max_structured_chars
    assert not prompt.endswith("…")


def test_structured_limit_drops_oldest_items_not_newest():
    summary = StructuredSummary(decisions=["cũ", "mới-1", "mới-2"])
    bounded = summary.limited(4_000, 2)
    assert bounded.decisions == ["mới-1", "mới-2"]


def test_progressive_batches_large_messages_and_merges_every_batch():
    messages = [{"role": "user", "content": "x" * 50_000} for _ in range(20)]
    # 20 messages split into at least two batches with a 10k input allowance.
    llm = MockLLMClient([text_response(_summary(decision=f"batch-{i}")) for i in range(20)])
    result = ProgressiveSummarizer(llm, max_input_chars=10_000).summarize(messages, None)
    assert len(llm.calls) > 1
    assert all(len(call["messages"][0]["content"]) <= 10_000 for call in llm.calls)
    assert result.decisions == [f"batch-{i}" for i in range(len(llm.calls))]


def test_progressive_json_wrappers_parse_and_invalid_json_falls_back_with_event():
    fenced = '```json\n{"goal":["giữ nguyên"],"decisions":[],"files_touched":[],"errors_encountered":[],"open_questions":[]}\n```'
    prose = 'đây là kết quả {"goal":[],"decisions":["ok"],"files_touched":[],"errors_encountered":[],"open_questions":[]} xong'
    progressive = ProgressiveSummarizer(MockLLMClient([text_response(fenced), text_response(prose)]))
    message = [{"role": "user", "content": "x"}]
    assert progressive.summarize(message, None).goal == ["giữ nguyên"]
    assert progressive.summarize(message, None).decisions == ["ok"]

    events: list[str] = []
    bus = EventBus()
    bus.subscribe("context.summarization_failed", lambda event: events.append(event.type))
    manager = SummarizingContextManager(
        keep_last_tool_results=1, event_bus=bus,
        summarizer=ProgressiveSummarizer(MockLLMClient([text_response('{"goal": [')]))
    )
    compacted = manager.compact(_state())
    assert "rút gọn" in str(compacted.messages[1]["content"])
    assert events == ["context.summarization_failed"]


def test_old_metadata_goal_round_trips_across_compactions():
    assert StructuredSummary.from_value({"decisions": [], "files_touched": [], "errors_encountered": [], "open_questions": []}) == StructuredSummary()
    old = StructuredSummary.from_value({"decisions": [], "files_touched": [], "errors_encountered": [], "open_questions": []})
    assert old is not None
    assert old.merged(StructuredSummary(goal=["yêu cầu nguyên văn"])).goal == ["yêu cầu nguyên văn"]


def test_provider_counter_has_bounded_hashed_cache_and_falls_back():
    class Unsupported(LLMClient):
        model = "test"
        def chat(self, messages, tools=None, system=None, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("unused")
        def count_tokens(self, text: str) -> int:
            raise NotImplementedError

    counter = ProviderTokenCounter(Unsupported(), fallback=lambda text: len(text), max_cache_entries=2)
    assert [counter(value) for value in ("a", "bb", "ccc")] == [1, 2, 3]
    assert len(counter._cache) == 2
    assert all(len(key) == 40 for key in counter._cache)
