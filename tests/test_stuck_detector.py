from kinetic_sdk.agent import Agent, StuckDetector
from kinetic_sdk.event import EventBus
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response
from kinetic_sdk.tool import ToolResult


def test_repeated_tool_call_stops_run_and_emits_event():
    bus = EventBus()
    events = []
    bus.subscribe("agent.stuck_detected", events.append)
    llm = MockLLMClient([tool_response(name="echo", arguments={"x": 1}), tool_response(name="echo", arguments={"x": 1}), tool_response(name="echo", arguments={"x": 1}), text_response("late")])
    agent = Agent(llm, tools=[MockTool("echo", result="ok")], event_bus=bus, permission_policy=PermissivePolicy(), stuck_detector=StuckDetector(window_size=3, repeat_threshold=3))
    assert agent.run("go") == ""
    assert len(llm.calls) == 3
    assert events[0].payload["tool_name"] == "echo"


def test_different_tool_arguments_do_not_trigger_detector():
    llm = MockLLMClient([tool_response(name="echo", arguments={"x": 1}), tool_response(name="echo", arguments={"x": 2}), tool_response(name="echo", arguments={"x": 3}), text_response("done")])
    agent = Agent(llm, tools=[MockTool("echo", result="ok")], permission_policy=PermissivePolicy(), stuck_detector=StuckDetector(window_size=3, repeat_threshold=3))
    assert agent.run("go") == "done"
    assert len(llm.calls) == 4


def test_changed_tool_result_does_not_trigger_detector():
    detector = StuckDetector(window_size=3, repeat_threshold=3)
    assert detector.check("echo", {"x": 1}, ToolResult(output="first")) is False
    assert detector.check("echo", {"x": 1}, ToolResult(output="second")) is False
    assert detector.check("echo", {"x": 1}, ToolResult(output="third")) is False
