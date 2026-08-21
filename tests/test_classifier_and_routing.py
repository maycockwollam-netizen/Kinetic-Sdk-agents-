"""Tests for the model-backed :class:`LiteLLMClassifier` and the agent's
FLASH/MAX routing + mid-run escalation.

Nothing here hits the network:

* The classifier is exercised through a fake :class:`LiteLLMClient` whose
  ``chat`` returns a scripted :class:`LLMResponse` (or raises) so we control the
  one-word reply and the failure path.
* The agent is exercised through :class:`MockLLM` plus a scriptable classifier
  (see :class:`_ScriptedClassifier`) that records how many times it was called.
"""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.classifier import (
    Classification,
    LiteLLMClassifier,
    TaskComplexity,
)
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.event.bus import EventBus, Event
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.llm.client import LLMResponse
from tests._helpers import EchoTool, FailingTool, MockLLM, text_response, tool_response


# --- helpers --------------------------------------------------------


class _FakeClassifierClient:
    """A stand-in for :class:`LiteLLMClient` used by the classifier.

    ``replies`` is a list of strings (the model's one-word answer) or
    exceptions to raise. Each ``chat`` call pops the next entry.
    """

    def __init__(self, replies: list[object]) -> None:
        self.model = "kinetic-classifier-v1"
        self._replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, system=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "system": system, "kwargs": kwargs})
        if not self._replies:
            return LLMResponse(content="COMPLEX", stop_reason="end_turn")
        entry = self._replies.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return LLMResponse(content=str(entry), stop_reason="end_turn")


class _ScriptedClassifier:
    """A classifier that returns a fixed complexity and counts its calls."""

    alias = "kinetic-classifier-v1"

    def __init__(self, complexity: TaskComplexity, raise_on_call: Exception | None = None) -> None:
        self._complexity = complexity
        self._raise = raise_on_call
        self.call_count = 0
        self.tasks: list[str] = []

    def classify(self, task: str) -> Classification:
        self.call_count += 1
        self.tasks.append(task)
        if self._raise is not None:
            raise self._raise
        return Classification(
            complexity=self._complexity,
            mode=self._complexity.to_mode(),
            confidence=0.9,
            rationale="test",
        )


# --- LiteLLMClassifier unit tests -----------------------------------


def test_classifier_parses_simple():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["SIMPLE"]))
    result = clf.classify("say hello")
    assert result.complexity is TaskComplexity.SIMPLE
    assert result.mode is AgentMode.FLASH


def test_classifier_parses_complex():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["COMPLEX"]))
    result = clf.classify("refactor the module")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX


def test_classifier_parses_mixed_case_and_whitespace():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["  complex  "]))
    result = clf.classify("do a thing")
    assert result.complexity is TaskComplexity.COMPLEX


def test_classifier_defaults_to_complex_on_unparseable_reply():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["maybe"]))
    result = clf.classify("do a thing")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX


def test_classifier_falls_back_to_complex_on_exception():
    clf = LiteLLMClassifier(client=_FakeClassifierClient([TimeoutError("boom")]))
    result = clf.classify("do a thing")
    assert result.complexity is TaskComplexity.COMPLEX
    assert result.mode is AgentMode.MAX


def test_classifier_max_tokens_is_low_and_no_tools():
    client = _FakeClassifierClient(["SIMPLE"])
    clf = LiteLLMClassifier(client=client)
    clf.classify("hi")
    assert client.calls
    sent = client.calls[0]["kwargs"]
    assert sent.get("max_tokens") == 10
    # The classifier must not expose tools to the model.
    assert client.calls[0]["tools"] is None


def test_classifier_prompt_requests_single_word():
    client = _FakeClassifierClient(["SIMPLE"])
    clf = LiteLLMClassifier(client=client)
    clf.classify("write a haiku")
    sent_messages = client.calls[0]["messages"]
    system = sent_messages[0]["content"]
    user = sent_messages[-1]["content"]
    assert "SIMPLE" in system and "COMPLEX" in system
    assert "write a haiku" in user


def test_classifier_summary_is_included_when_provided():
    client = _FakeClassifierClient(["SIMPLE"])
    clf = LiteLLMClassifier(client=client, summary="prior turns happened")
    clf.classify("continue")
    user = client.calls[0]["messages"][-1]["content"]
    assert "prior turns happened" in user


def test_classifier_public_model_is_alias_only():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["SIMPLE"]))
    assert clf.model == "kinetic-classifier-v1"
    assert clf.alias == "kinetic-classifier-v1"


def test_classifier_never_leaks_real_model_in_rationale():
    clf = LiteLLMClassifier(client=_FakeClassifierClient(["SIMPLE"]))
    result = clf.classify("hi")
    assert "glm-5.2" not in result.rationale
    assert "llm-proxy" not in result.rationale


# --- Agent routing tests --------------------------------------------


def test_run_routes_to_flash_when_simple():
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    llm = MockLLM([text_response("ok")])
    agent = Agent(llm=llm, tools=[], classifier=clf)
    agent.run("hi")
    assert agent.mode is AgentMode.FLASH
    assert agent.enable_extended_reasoning is False
    assert agent.max_iterations == Agent.MODE_MAX_ITERATIONS[AgentMode.FLASH]
    assert clf.call_count == 1


def test_run_routes_to_max_when_complex():
    clf = _ScriptedClassifier(TaskComplexity.COMPLEX)
    llm = MockLLM([text_response("ok")])
    agent = Agent(llm=llm, tools=[], classifier=clf)
    agent.run("refactor everything")
    assert agent.mode is AgentMode.MAX
    assert agent.enable_extended_reasoning is True
    assert agent.max_iterations == Agent.MODE_MAX_ITERATIONS[AgentMode.MAX]
    assert clf.call_count == 1


def test_run_falls_back_to_max_when_classifier_raises():
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE, raise_on_call=RuntimeError("net down"))
    llm = MockLLM([text_response("ok")])
    events: list[Event] = []
    agent = Agent(llm=llm, tools=[], classifier=clf, event_bus=EventBus())
    agent.event_bus.subscribe("agent.classified", events.append)
    agent.run("hi")
    # Fallback: MAX, no crash, no classified event (classification failed).
    assert agent.mode is AgentMode.MAX
    assert agent.enable_extended_reasoning is True
    assert events == []


def test_flash_caps_tool_iterations():
    # Model keeps calling echo forever; FLASH escalates at the iteration
    # threshold (3) -> MAX (50), so the run continues under the MAX cap.
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    loop = tool_response("c", "echo", {"message": "x"})
    llm = MockLLM([loop] * 100)
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe("agent.error", events.append)
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy(), classifier=clf, event_bus=bus)
    agent.run("loop forever")
    assert any(e.payload.get("reason") == "max_iterations" for e in events)
    assert agent.mode is AgentMode.MAX
    assert agent.max_iterations == Agent.MODE_MAX_ITERATIONS[AgentMode.MAX]


def test_escalation_on_first_tool_error():
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    # Turn 1 (FLASH): call the failing tool -> error -> escalate to MAX.
    # Turn 2 (MAX): final answer.
    llm = MockLLM(
        [
            tool_response("c1", "boom", {}),
            text_response("recovered in MAX"),
        ]
    )
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe("agent.escalated", events.append)
    agent = Agent(llm=llm, tools=[FailingTool()], permission_policy=PermissivePolicy(), classifier=clf, event_bus=bus)
    result = agent.run("trigger failure")
    assert result == "recovered in MAX"
    assert agent.mode is AgentMode.MAX
    assert agent.enable_extended_reasoning is True
    # Escalated exactly once.
    assert len(events) == 1
    assert events[0].payload["from"] == "flash"
    assert events[0].payload["to"] == "max"


def test_escalation_on_threshold_without_tool_error():
    # Model keeps calling a healthy echo; FLASH escalates at the iteration
    # threshold (3) without any tool error, then finishes in MAX.
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": "1"}),
            tool_response("c2", "echo", {"message": "2"}),
            tool_response("c3", "echo", {"message": "3"}),
            text_response("done after escalation"),
        ]
    )
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe("agent.escalated", events.append)
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy(), classifier=clf, event_bus=bus)
    result = agent.run("go")
    assert result == "done after escalation"
    assert agent.mode is AgentMode.MAX
    assert len(events) == 1


def test_no_escalation_when_already_max():
    clf = _ScriptedClassifier(TaskComplexity.COMPLEX)
    llm = MockLLM([tool_response("c1", "boom", {}), text_response("ok")])
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe("agent.escalated", events.append)
    agent = Agent(llm=llm, tools=[FailingTool()], permission_policy=PermissivePolicy(), classifier=clf, event_bus=bus)
    agent.run("go")
    assert agent.mode is AgentMode.MAX
    assert events == []


def test_classifier_called_exactly_once_even_with_escalation():
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    llm = MockLLM(
        [
            tool_response("c1", "boom", {}),
            text_response("ok"),
        ]
    )
    agent = Agent(llm=llm, tools=[FailingTool()], permission_policy=PermissivePolicy(), classifier=clf)
    agent.run("go")
    assert clf.call_count == 1


def test_no_downgrade_from_max_to_flash():
    clf = _ScriptedClassifier(TaskComplexity.COMPLEX)
    llm = MockLLM([text_response("ok")])
    agent = Agent(llm=llm, tools=[], classifier=clf)
    agent.run("go")
    assert agent.mode is AgentMode.MAX
    # An explicit escalate() on MAX must be a no-op.
    assert agent.escalate() is False
    assert agent.mode is AgentMode.MAX


def test_user_override_of_max_iterations_respected_initially():
    clf = _ScriptedClassifier(TaskComplexity.SIMPLE)
    loop = tool_response("c", "echo", {"message": "x"})
    llm = MockLLM([loop] * 100)
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe("agent.error", events.append)
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=PermissivePolicy(), classifier=clf, event_bus=bus, max_iterations=2)
    agent.run("go")
    # Override pins the cap at 2 even though FLASH default is 5; the threshold
    # escalation (3) is never reached because the cap is 2 -> max_iterations
    # error in FLASH (no escalation).
    assert agent.mode is AgentMode.FLASH
    assert any(e.payload.get("reason") == "max_iterations" for e in events)
