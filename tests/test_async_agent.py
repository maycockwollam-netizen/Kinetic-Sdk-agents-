"""Tests for the async agent loop (``AsyncAgent``).

Mirrors the sync ``tests/test_agent.py`` coverage: basic runs, tool calling,
events, hooks, permission policy + confirmation, escalation, cancellation,
streaming, persistence, validation, timeouts and parallel execution — all
driven by the scripted ``AsyncMockLLMClient``/``AsyncMockTool`` fakes.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kinetic_sdk.agent.async_agent import AsyncAgent
from kinetic_sdk.agent.async_classifier import AsyncTaskClassifier
from kinetic_sdk.agent.classifier import (
    Classification,
    DefaultClassifier,
    TaskComplexity,
)
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.context.manager import NoopContextManager
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import JsonFileConversationStore
from kinetic_sdk.event.bus import EventBus
from kinetic_sdk.hooks.base import HookContext, HookPoint, HookResult
from kinetic_sdk.hooks.registry import HookRegistry
from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.observability.logger import InMemoryObservabilityLogger
from kinetic_sdk.observability.trace import RunTrace
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.security.policy import AllowListPolicy, PermissivePolicy
from kinetic_sdk.testing import (
    AsyncMockLLMClient,
    AsyncMockTool,
    MockLLMClient,
    MockTool,
    assert_mode,
    assert_no_permission_denied,
    assert_tool_called,
    text_response,
    tool_response,
)
from kinetic_sdk.tool.base import ToolResult

pytestmark = pytest.mark.asyncio


def make_agent(llm, tools=(), **kwargs):
    kwargs.setdefault("permission_policy", PermissivePolicy())
    return AsyncAgent(llm=llm, tools=tools, **kwargs)


# --- basic loop -------------------------------------------------------


async def test_simple_text_run():
    llm = AsyncMockLLMClient([text_response("Xin chào!")])
    agent = make_agent(llm)
    result = await agent.run("hello")
    assert result == "Xin chào!"
    assert len(llm.calls) == 1


async def test_tool_call_then_final_answer():
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "hi"}),
        text_response("The echo said hi."),
    ])
    echo = AsyncMockTool("echo", result="hi")
    agent = make_agent(llm, tools=[echo])
    result = await agent.run("say hi")
    assert result == "The echo said hi."
    assert echo.calls == [{"message": "hi"}]
    # History: user, assistant(tool_use), user(tool_result), assistant(text)
    assert len(agent.state.messages) == 4


async def test_run_without_message_continues_conversation():
    llm = AsyncMockLLMClient([text_response("first"), text_response("second")])
    agent = make_agent(llm)
    assert await agent.run("first") == "first"
    result = await agent.run()
    assert result == "second"
    # No new user message appended on the second run.
    user_msgs = [m for m in agent.state.messages if m.get("role") == "user"]
    assert len(user_msgs) == 1


async def test_max_iterations_stops_loop():
    # Model that never stops calling tools.
    entries = [tool_response(f"c{i}", "echo", {"message": "x"}) for i in range(10)]
    llm = AsyncMockLLMClient(entries)
    echo = AsyncMockTool("echo", result="x")
    agent = make_agent(llm, tools=[echo], max_iterations=3)
    result = await agent.run("loop")
    assert result == ""
    assert len(echo.calls) == 3


async def test_max_iterations_emits_error_event():
    llm = AsyncMockLLMClient([tool_response("c1", "echo", {"message": "x"})] * 5)
    bus = EventBus()
    seen = []
    bus.subscribe("agent.error", seen.append)
    agent = make_agent(
        llm, tools=[AsyncMockTool("echo", result="x")], event_bus=bus, max_iterations=2
    )
    await agent.run("loop")
    assert any(e.payload.get("reason") == "max_iterations" for e in seen)


async def test_unknown_tool_returns_error_to_model():
    llm = AsyncMockLLMClient([
        tool_response("c1", "ghost", {}),
        text_response("recovered"),
    ])
    agent = make_agent(llm)
    result = await agent.run("hi")
    assert result == "recovered"
    tool_result_msg = agent.state.messages[2]
    block = tool_result_msg["content"][0]
    assert block["is_error"] is True
    assert "Unknown tool" in block["content"]


async def test_tool_exception_becomes_error_result():
    async def boom(**params):
        raise RuntimeError("kaboom")

    llm = AsyncMockLLMClient([
        tool_response("c1", "boom", {}),
        text_response("handled"),
    ])
    tool = AsyncMockTool("boom", handler=boom)
    agent = make_agent(llm, tools=[tool])
    result = await agent.run("hi")
    assert result == "handled"
    block = agent.state.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "kaboom" in block["content"]


async def test_duplicate_tool_name_rejected():
    with pytest.raises(ValueError, match="Duplicate tool name"):
        make_agent(AsyncMockLLMClient([]), tools=[AsyncMockTool("x"), AsyncMockTool("x")])


async def test_add_tool_at_runtime():
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]))
    agent.add_tool(AsyncMockTool("late"))
    assert "late" in [s["name"] for s in agent.tool_schemas()]
    with pytest.raises(ValueError, match="Duplicate tool name"):
        agent.add_tool(AsyncMockTool("late"))


async def test_run_id_and_run_started_event():
    bus = EventBus()
    seen = []
    bus.subscribe("agent.run_started", seen.append)
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]), event_bus=bus)
    await agent.run("hi")
    assert agent.run_id is not None
    assert seen[0].payload["run_id"] == agent.run_id
    assert seen[0].payload["mode"] == "max"


# --- LLM / classifier input flexibility --------------------------------


async def test_sync_llm_client_is_auto_wrapped():
    llm = MockLLMClient([text_response("sync-wrapped")])
    agent = make_agent(llm)
    result = await agent.run("hi")
    assert result == "sync-wrapped"
    assert len(llm.calls) == 1


async def test_sync_classifier_is_bridged():
    llm = AsyncMockLLMClient([text_response("ok")])
    agent = make_agent(llm, classifier=DefaultClassifier())
    assert isinstance(agent.classifier, AsyncTaskClassifier)
    result = await agent.run("hi")
    assert result == "ok"
    assert agent.mode is AgentMode.MAX


async def test_invalid_llm_type_rejected():
    with pytest.raises(TypeError, match="AsyncLLMClient or LLMClient"):
        AsyncAgent(llm=object())


async def test_invalid_classifier_type_rejected():
    with pytest.raises(TypeError, match="AsyncTaskClassifier or TaskClassifier"):
        make_agent(AsyncMockLLMClient([]), classifier=object())


# --- routing / escalation -----------------------------------------------


class ScriptedClassifier(AsyncTaskClassifier):
    def __init__(self, *classifications):
        self._items = list(classifications)
        self.calls = 0

    async def classify(self, task):
        self.calls += 1
        return self._items.pop(0)


def simple_classification():
    return Classification(
        complexity=TaskComplexity.SIMPLE,
        mode=AgentMode.FLASH,
        confidence=0.9,
        rationale="test",
    )


def complex_classification():
    return Classification(
        complexity=TaskComplexity.COMPLEX,
        mode=AgentMode.MAX,
        confidence=0.9,
        rationale="test",
    )


async def test_flash_mode_routing_caps_iterations():
    llm = AsyncMockLLMClient([text_response("quick")])
    agent = make_agent(llm, classifier=ScriptedClassifier(simple_classification()))
    await agent.run("easy")
    assert agent.mode is AgentMode.FLASH
    assert agent.max_iterations == AsyncAgent.MODE_MAX_ITERATIONS[AgentMode.FLASH]
    assert agent.enable_extended_reasoning is False


async def test_flash_escalates_on_first_turn_tool_error():
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("recovered"),
    ])
    echo = AsyncMockTool("echo", result=ToolResult(error="broken"))
    bus = EventBus()
    events = []
    bus.subscribe("agent.escalated", events.append)
    agent = make_agent(
        llm,
        tools=[echo],
        classifier=ScriptedClassifier(simple_classification()),
        event_bus=bus,
    )
    result = await agent.run("do it")
    assert result == "recovered"
    assert agent.mode is AgentMode.MAX
    assert agent.max_iterations == AsyncAgent.MODE_MAX_ITERATIONS[AgentMode.MAX]
    assert len(events) == 1
    assert events[0].payload["from"] == "flash"
    assert events[0].payload["to"] == "max"


async def test_flash_escalates_after_threshold_iterations():
    entries = [tool_response(f"c{i}", "echo", {"message": "x"}) for i in range(3)]
    entries.append(text_response("finally"))
    llm = AsyncMockLLMClient(entries)
    echo = AsyncMockTool("echo", result="fine")
    agent = make_agent(llm, tools=[echo], classifier=ScriptedClassifier(simple_classification()))
    result = await agent.run("work")
    assert result == "finally"
    assert agent.mode is AgentMode.MAX


async def test_max_never_downgrades_to_flash():
    llm = AsyncMockLLMClient([text_response("one"), text_response("two")])
    classifier = ScriptedClassifier(complex_classification(), simple_classification())
    agent = make_agent(llm, classifier=classifier)
    await agent.run("first")
    assert agent.mode is AgentMode.MAX
    await agent.run("second")
    # Sticky MAX: the second (SIMPLE) classification must not downgrade.
    assert agent.mode is AgentMode.MAX
    assert agent.max_iterations == AsyncAgent.MODE_MAX_ITERATIONS[AgentMode.MAX]


async def test_classifier_exception_falls_back_to_max():
    class ExplodingClassifier(AsyncTaskClassifier):
        async def classify(self, task):
            raise RuntimeError("classifier down")

    llm = AsyncMockLLMClient([text_response("ok")])
    agent = make_agent(llm, classifier=ExplodingClassifier())
    result = await agent.run("hi")
    assert result == "ok"
    assert agent.mode is AgentMode.MAX
    # Exception fallback must NOT pin sticky MAX.
    assert agent._sticky_max is False


async def test_escalate_returns_false_at_max():
    agent = make_agent(AsyncMockLLMClient([]))
    assert agent.mode is AgentMode.MAX
    assert agent.escalate() is False


# --- hooks ---------------------------------------------------------------


async def test_async_hooks_are_awaited():
    seen = []

    async def before_run(ctx: HookContext):
        seen.append(("before_run", ctx.user_message))

    async def after_run(ctx: HookContext):
        seen.append(("after_run", ctx.final_text))

    hooks = HookRegistry()
    hooks.register(HookPoint.BEFORE_RUN, before_run)
    hooks.register(HookPoint.AFTER_RUN, after_run)
    agent = make_agent(AsyncMockLLMClient([text_response("done")]), hooks=hooks)
    await agent.run("hello")
    assert seen == [("before_run", "hello"), ("after_run", "done")]


async def test_sync_hooks_still_work_in_async_loop():
    seen = []
    hooks = HookRegistry()
    hooks.register(HookPoint.BEFORE_LLM_CALL, lambda ctx: seen.append(ctx.iteration))
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]), hooks=hooks)
    await agent.run("hi")
    assert seen == [0]


async def test_before_tool_call_hook_cancels_call():
    hooks = HookRegistry()
    hooks.register(
        HookPoint.BEFORE_TOOL_CALL,
        lambda ctx: HookResult(should_continue=False),
    )
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("cancelled path"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = make_agent(llm, tools=[echo], hooks=hooks)
    result = await agent.run("hi")
    assert result == "cancelled path"
    assert echo.calls == []
    block = agent.state.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "cancelled by a before_tool_call hook" in block["content"]


async def test_before_tool_call_hook_replaces_input():
    async def rewrite(ctx: HookContext):
        return HookResult(modified_context={"tool_input": {"message": "rewritten"}})

    hooks = HookRegistry()
    hooks.register(HookPoint.BEFORE_TOOL_CALL, rewrite)
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "original"}),
        text_response("ok"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = make_agent(llm, tools=[echo], hooks=hooks)
    await agent.run("hi")
    assert echo.calls == [{"message": "rewritten"}]


async def test_raising_hook_does_not_crash_and_emits_hooks_error():
    async def bad_hook(ctx: HookContext):
        raise RuntimeError("hook exploded")

    bus = EventBus()
    errors = []
    bus.subscribe("hooks.error", errors.append)
    hooks = HookRegistry()
    hooks.register(HookPoint.BEFORE_RUN, bad_hook)
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]), hooks=hooks, event_bus=bus)
    result = await agent.run("hi")
    assert result == "ok"
    assert len(errors) == 1
    assert "hook exploded" in errors[0].payload["error"]


async def test_on_error_hook_fires_on_exception():
    class ExplodingLLM(AsyncMockLLMClient):
        async def chat(self, *args, **kwargs):
            raise RuntimeError("llm down")

    seen = []
    hooks = HookRegistry()
    hooks.register(HookPoint.ON_ERROR, lambda ctx: seen.append(ctx.error))
    agent = make_agent(ExplodingLLM([]), hooks=hooks)
    with pytest.raises(RuntimeError, match="llm down"):
        await agent.run("hi")
    assert seen and "llm down" in seen[0]


# --- permission policy / confirmation ------------------------------------


async def test_default_policy_denies_everything():
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("denied path"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = AsyncAgent(llm=llm, tools=[echo])  # default AllowListPolicy()
    result = await agent.run("hi")
    assert result == "denied path"
    assert echo.calls == []
    block = agent.state.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "Permission denied" in block["content"]


async def test_permission_denied_event_and_audit():
    bus = EventBus()
    denials = []
    bus.subscribe("security.permission_denied", denials.append)
    audit = InMemoryAuditLogger()
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("ok"),
    ])
    agent = AsyncAgent(llm=llm, tools=[AsyncMockTool("echo")], event_bus=bus, audit_logger=audit)
    await agent.run("hi")
    assert len(denials) == 1
    assert denials[0].payload["name"] == "echo"
    assert any(e["event"] == "permission_denied" for e in audit.entries)


async def test_requires_confirmation_denied_without_hooks():
    policy = AllowListPolicy(
        always_allow=["echo"],
        require_confirmation_patterns={"echo": [".*"]},
    )
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("ok"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = AsyncAgent(llm=llm, tools=[echo], permission_policy=policy)
    await agent.run("hi")
    assert echo.calls == []
    block = agent.state.messages[2]["content"][0]
    assert "requires manual confirmation" in block["content"]


async def test_requires_confirmation_approved_by_async_hook():
    async def confirm(ctx: HookContext):
        return HookResult(should_continue=True)

    hooks = HookRegistry()
    hooks.register(HookPoint.ON_PERMISSION_CHECK, confirm)
    policy = AllowListPolicy(
        always_allow=["echo"],
        require_confirmation_patterns={"echo": [".*"]},
    )
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("ok"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = AsyncAgent(llm=llm, tools=[echo], permission_policy=policy, hooks=hooks)
    await agent.run("hi")
    assert echo.calls == [{"message": "x"}]


async def test_after_tool_call_hook_skipped_on_denial():
    seen = []
    hooks = HookRegistry()
    hooks.register(HookPoint.AFTER_TOOL_CALL, lambda ctx: seen.append(ctx.tool_name))
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "x"}),
        text_response("ok"),
    ])
    agent = AsyncAgent(llm=llm, tools=[AsyncMockTool("echo")], hooks=hooks)
    await agent.run("hi")
    assert seen == []


# --- schema validation ----------------------------------------------------


async def test_invalid_tool_input_rejected_with_actionable_message():
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"wrong_field": 1}),
        text_response("ok"),
    ])
    echo = AsyncMockTool(
        "echo",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        result="x",
    )
    agent = make_agent(llm, tools=[echo])
    await agent.run("hi")
    assert echo.calls == []
    block = agent.state.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "invalid tool input" in block["content"]


async def test_validation_can_be_disabled():
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"anything": "goes"}),
        text_response("ok"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = make_agent(llm, tools=[echo], validate_tool_inputs=False)
    await agent.run("hi")
    assert echo.calls == [{"anything": "goes"}]


# --- streaming ------------------------------------------------------------


async def test_streaming_emits_text_deltas():
    bus = EventBus()
    deltas = []
    bus.subscribe("agent.text_delta", lambda e: deltas.append(e.payload["delta"]))
    llm = AsyncMockLLMClient([text_response("streaming answer here")])
    agent = make_agent(llm, event_bus=bus)
    result = await agent.run("hi", stream=True)
    assert result == "streaming answer here"
    assert "".join(deltas) == "streaming answer here"
    assert len(deltas) > 1  # chunked, like a real provider


async def test_streaming_with_tool_calls_still_works():
    bus = EventBus()
    deltas = []
    bus.subscribe("agent.text_delta", lambda e: deltas.append(e.payload["delta"]))
    llm = AsyncMockLLMClient([
        LLMResponse(
            content="let me check",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])
    agent = make_agent(llm, event_bus=bus)
    result = await agent.run("hi", stream=True)
    assert result == "let me check"
    assert "".join(deltas) == "let me check"


async def test_streaming_falls_back_when_unsupported():
    class NoStreamLLM(AsyncMockLLMClient):
        async def chat_stream(self, *args, **kwargs):
            raise NotImplementedError("no streaming")
            yield  # pragma: no cover

    llm = NoStreamLLM([text_response("fallback answer")])
    agent = make_agent(llm)
    result = await agent.run("hi", stream=True)
    assert result == "fallback answer"


# --- cancellation ----------------------------------------------------------


async def test_cancel_between_iterations():
    call_count = 0

    async def counting_handler(**params):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)  # give the canceller a real window
        return "x"

    llm = AsyncMockLLMClient(
        [tool_response(f"c{i}", "echo", {"message": "x"}) for i in range(10)]
    )
    echo = AsyncMockTool("echo", handler=counting_handler)
    bus = EventBus()
    cancelled_events = []
    bus.subscribe("agent.cancelled", cancelled_events.append)
    agent = make_agent(llm, tools=[echo], event_bus=bus)

    async def cancel_soon():
        await asyncio.sleep(0.05)
        agent.cancel()

    task = asyncio.create_task(cancel_soon())
    result = await agent.run("loop")
    await task
    assert result == ""
    assert cancelled_events
    assert call_count < 10


async def test_cancel_mid_batch_skips_remaining_calls():
    llm = AsyncMockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                tool_response("c1", "echo", {"message": "1"}).tool_calls[0],
                tool_response("c2", "echo", {"message": "2"}).tool_calls[0],
            ],
            stop_reason="tool_use",
        ),
        text_response("done"),
    ])

    async def cancel_on_first(**params):
        agent.cancel()
        return "x"

    echo = AsyncMockTool("echo", handler=cancel_on_first)
    agent = make_agent(llm, tools=[echo])
    result = await agent.run("hi")
    # Second call skipped but still recorded as an error tool_result so the
    # history stays replay-valid. Each result is its own user turn.
    assert echo.calls == [{"message": "1"}]
    block = agent.state.messages[3]["content"][0]
    assert block["is_error"] is True
    assert "skipped: run cancelled" in block["content"]
    assert result == ""


async def test_cancel_flag_cleared_on_new_run():
    llm = AsyncMockLLMClient([text_response("ok")])
    agent = make_agent(llm)
    agent.cancel()
    assert agent.cancelled is True
    await agent.run("hi")
    assert agent.cancelled is False


# --- observability / trace -------------------------------------------------


async def test_observability_logger_captures_full_run():
    obs = InMemoryObservabilityLogger()
    llm = AsyncMockLLMClient([
        tool_response("c1", "echo", {"message": "hi"}),
        text_response("final"),
    ])
    echo = AsyncMockTool("echo", result="hi")
    agent = make_agent(llm, tools=[echo], observability_logger=obs)
    await agent.run("hi")
    trace = RunTrace.collect(obs.entries, agent.run_id)
    assert_tool_called(trace, "echo", times=1)
    assert_mode(trace, "max")
    assert_no_permission_denied(trace)
    summary = trace.to_summary()
    assert summary["tool_call_count"] == 1
    assert summary["tool_calls_failed"] == 0


async def test_coroutine_event_subscriber_is_awaited():
    bus = EventBus()
    seen = []

    async def async_listener(event):
        await asyncio.sleep(0)
        seen.append(event.type)

    bus.subscribe("agent.run_finished", async_listener)
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]), event_bus=bus)
    await agent.run("hi")
    assert seen == ["agent.run_finished"]


# --- persistence -----------------------------------------------------------


async def test_state_persisted_and_resumed(tmp_path):
    store = JsonFileConversationStore(tmp_path / "conv.json")
    llm = AsyncMockLLMClient([text_response("first answer")])
    agent = make_agent(llm, state_store=store)
    await agent.run("hello")
    assert (tmp_path / "conv.json").exists()

    llm2 = AsyncMockLLMClient([text_response("second answer")])
    agent2 = make_agent(llm2, state_store=store)
    # Resumed history includes the first run's messages.
    assert any(
        m.get("content") == "hello" for m in agent2.state.messages if m.get("role") == "user"
    )
    result = await agent2.run("again")
    assert result == "second answer"


async def test_broken_store_never_kills_run(tmp_path):
    class BrokenStore(JsonFileConversationStore):
        def save(self, state):
            raise OSError("disk full")

    bus = EventBus()
    failures = []
    bus.subscribe("agent.state_persist_failed", failures.append)
    store = BrokenStore(tmp_path / "conv.json")
    agent = make_agent(AsyncMockLLMClient([text_response("ok")]), state_store=store, event_bus=bus)
    result = await agent.run("hi")
    assert result == "ok"
    assert failures


# --- context compaction ----------------------------------------------------


async def test_compaction_event_emitted():
    from kinetic_sdk.context.manager import SimpleTruncateContextManager

    manager = SimpleTruncateContextManager(keep_last_tool_results=1, safety_threshold=0.5)
    bus = EventBus()
    compacted = []
    bus.subscribe("context.compacted", compacted.append)
    state = ConversationState()
    for i in range(6):
        state.add_user_message(f"message {i} " + "x" * 200)
        state.add_assistant(f"answer {i}")
    llm = AsyncMockLLMClient([text_response("ok")])
    agent = make_agent(
        llm,
        state=state,
        context_manager=manager,
        model_context_limit=100,
        event_bus=bus,
    )
    await agent.run("one more")
    assert compacted
    assert compacted[0].payload["messages_removed"] > 0


async def test_noop_context_manager_disables_compaction():
    state = ConversationState()
    for i in range(6):
        state.add_user_message("x" * 500)
    llm = AsyncMockLLMClient([text_response("ok")])
    agent = make_agent(
        llm, state=state, context_manager=NoopContextManager(), model_context_limit=10
    )
    await agent.run("hi")
    # Nothing elided: all original messages plus the new turn remain.
    assert len(agent.state.messages) >= 8


# --- tool timeout ------------------------------------------------------------


async def test_tool_timeout_returns_error_result():
    async def slow(**params):
        await asyncio.sleep(5)
        return "too late"

    llm = AsyncMockLLMClient([
        tool_response("c1", "slow", {}),
        text_response("timed out, moving on"),
    ])
    tool = AsyncMockTool("slow", handler=slow)
    agent = make_agent(llm, tools=[tool], tool_timeout=0.05)
    result = await agent.run("hi")
    assert result == "timed out, moving on"
    block = agent.state.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "timed out" in block["content"]


async def test_tool_timeout_cancels_native_async_tool():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(**params):
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "late"

    llm = AsyncMockLLMClient([
        tool_response("c1", "slow", {}),
        text_response("ok"),
    ])
    tool = AsyncMockTool("slow", handler=slow)
    agent = make_agent(llm, tools=[tool], tool_timeout=0.05)
    await agent.run("hi")
    assert started.is_set()
    # Native-async tools are cancelled for real on timeout.
    assert cancelled.is_set()


async def test_invalid_tool_timeout_rejected():
    with pytest.raises(ValueError, match="tool_timeout must be positive"):
        make_agent(AsyncMockLLMClient([]), tool_timeout=0)


# --- parallel tool execution -------------------------------------------------


async def test_parallel_execution_runs_calls_concurrently():
    started: list[str] = []
    release = asyncio.Event()

    async def gated(**params):
        started.append(params["message"])
        await release.wait()
        return params["message"]

    llm = AsyncMockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                tool_response("c1", "echo", {"message": "a"}).tool_calls[0],
                tool_response("c2", "echo", {"message": "b"}).tool_calls[0],
            ],
            stop_reason="tool_use",
        ),
        text_response("both done"),
    ])
    echo = AsyncMockTool("echo", handler=gated)
    agent = make_agent(llm, tools=[echo], parallel_tool_execution=True)

    async def release_soon():
        # If execution were sequential, the first call would block forever
        # waiting for the release that only this helper provides.
        await asyncio.sleep(0.05)
        release.set()

    helper = asyncio.create_task(release_soon())
    result = await asyncio.wait_for(agent.run("hi"), timeout=2)
    await helper
    assert result == "both done"
    assert sorted(started) == ["a", "b"]


async def test_parallel_results_finalized_in_model_order():
    async def delayed(**params):
        # "slow" finishes last even though it was called first.
        await asyncio.sleep(0.05 if params["message"] == "slow" else 0)
        return params["message"]

    llm = AsyncMockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                tool_response("c1", "echo", {"message": "slow"}).tool_calls[0],
                tool_response("c2", "echo", {"message": "fast"}).tool_calls[0],
            ],
            stop_reason="tool_use",
        ),
        text_response("done"),
    ])
    echo = AsyncMockTool("echo", handler=delayed)
    agent = make_agent(llm, tools=[echo], parallel_tool_execution=True)
    await agent.run("hi")
    # Each tool_result is its own user turn; order must follow the model's
    # call order, not completion order.
    contents = [
        agent.state.messages[2]["content"][0]["content"],
        agent.state.messages[3]["content"][0]["content"],
    ]
    assert contents == ["slow", "fast"]


async def test_parallel_gating_still_enforced():
    # Default deny policy must block calls even in parallel mode.
    llm = AsyncMockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                tool_response("c1", "echo", {"message": "a"}).tool_calls[0],
                tool_response("c2", "echo", {"message": "b"}).tool_calls[0],
            ],
            stop_reason="tool_use",
        ),
        text_response("ok"),
    ])
    echo = AsyncMockTool("echo", result="x")
    agent = AsyncAgent(llm=llm, tools=[echo], parallel_tool_execution=True)
    await agent.run("hi")
    assert echo.calls == []
    blocks = [
        agent.state.messages[2]["content"][0],
        agent.state.messages[3]["content"][0],
    ]
    assert all(b["is_error"] for b in blocks)


async def test_parallel_tool_error_does_not_sweep_others():
    async def maybe_boom(**params):
        if params["message"] == "bad":
            raise RuntimeError("kaboom")
        return "fine"

    llm = AsyncMockLLMClient([
        LLMResponse(
            content="",
            tool_calls=[
                tool_response("c1", "echo", {"message": "bad"}).tool_calls[0],
                tool_response("c2", "echo", {"message": "good"}).tool_calls[0],
            ],
            stop_reason="tool_use",
        ),
        text_response("done"),
    ])
    echo = AsyncMockTool("echo", handler=maybe_boom)
    agent = make_agent(llm, tools=[echo], parallel_tool_execution=True)
    result = await agent.run("hi")
    assert result == "done"
    blocks = [
        agent.state.messages[2]["content"][0],
        agent.state.messages[3]["content"][0],
    ]
    assert blocks[0]["is_error"] is True
    assert "kaboom" in blocks[0]["content"]
    assert blocks[1]["content"] == "fine"


# --- sync tools under the async loop ----------------------------------------


async def test_sync_tool_works_via_default_execute_async():
    llm = AsyncMockLLMClient([
        tool_response("c1", "sync_tool", {"value": 3}),
        text_response("ok"),
    ])
    tool = MockTool("sync_tool", handler=lambda value: value * 2)
    agent = make_agent(llm, tools=[tool])
    result = await agent.run("hi")
    assert result == "ok"
    assert tool.calls == [{"value": 3}]
    block = agent.state.messages[2]["content"][0]
    assert json.loads(block["content"]) == 6


async def test_sync_tool_does_not_block_event_loop():
    import time

    llm = AsyncMockLLMClient([
        tool_response("c1", "sync_tool", {}),
        text_response("ok"),
    ])
    tool = MockTool("sync_tool", handler=lambda: time.sleep(0.1) or "slept")
    agent = make_agent(llm, tools=[tool])

    ticked = asyncio.Event()

    async def ticker():
        while not ticked.is_set():
            ticked.set()
            await asyncio.sleep(0.01)

    helper = asyncio.create_task(ticker())
    await agent.run("hi")
    await asyncio.wait_for(ticked.wait(), timeout=1)
    helper.cancel()


# --- true concurrency: independent agents in parallel ------------------------


async def test_independent_agents_run_concurrently():
    """The headline async feature: N agents overlap their I/O waits."""

    async def slow_llm_reply(messages, tools, system):
        await asyncio.sleep(0.1)  # simulated provider latency
        return text_response("done")

    agents = [
        make_agent(AsyncMockLLMClient([slow_llm_reply])) for _ in range(5)
    ]
    start = asyncio.get_event_loop().time()
    results = await asyncio.gather(*(a.run(f"task {i}") for i, a in enumerate(agents)))
    elapsed = asyncio.get_event_loop().time() - start
    assert results == ["done"] * 5
    # Sequential execution would take >= 0.5s; concurrent ~= one latency.
    assert elapsed < 0.4


async def test_concurrent_agents_keep_separate_state_and_run_ids():
    agents = [
        make_agent(AsyncMockLLMClient([text_response(f"answer {i}")]))
        for i in range(3)
    ]
    results = await asyncio.gather(*(a.run(f"question {i}") for i, a in enumerate(agents)))
    assert results == [f"answer {i}" for i in range(3)]
    run_ids = {a.run_id for a in agents}
    assert len(run_ids) == 3
    for i, a in enumerate(agents):
        assert a.state.messages[0]["content"] == f"question {i}"
