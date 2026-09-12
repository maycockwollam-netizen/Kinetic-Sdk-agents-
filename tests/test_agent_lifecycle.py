"""Tests for cooperative pause/resume and in-flight message injection."""

from __future__ import annotations

import threading
import time

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response


def test_pause_holds_after_a_tool_round_until_resume():
    started = threading.Event()
    tool = MockTool(name="slow", handler=lambda: (started.set(), time.sleep(0.05), "done")[2])
    agent = Agent(
        llm=MockLLMClient([tool_response(name="slow"), text_response("finished")]),
        tools=[tool], permission_policy=PermissivePolicy(),
    )
    result: list[str] = []
    thread = threading.Thread(target=lambda: result.append(agent.run("go")))
    thread.start()
    assert started.wait(1)
    agent.pause()
    time.sleep(0.1)
    assert thread.is_alive()
    assert len(agent.state.messages) == 3  # user, tool_use assistant, tool_result
    agent.resume()
    thread.join(1)
    assert result == ["finished"]


def test_message_injected_while_running_is_seen_by_next_llm_turn():
    started = threading.Event()
    tool = MockTool(name="slow", handler=lambda: (started.set(), time.sleep(0.05), "done")[2])

    def final_from_last_user(messages, _tools, _system):
        users = [message["content"] for message in messages if message["role"] == "user" and isinstance(message["content"], str)]
        return text_response("ack: " + users[-1])

    agent = Agent(
        llm=MockLLMClient([tool_response(name="slow"), final_from_last_user]),
        tools=[tool], permission_policy=PermissivePolicy(),
    )
    seen = []
    agent.event_bus.subscribe("agent.message_injected", lambda event: seen.append(event.payload["message"]))
    result: list[str] = []
    thread = threading.Thread(target=lambda: result.append(agent.run("original")))
    thread.start()
    assert started.wait(1)
    agent.send_message_while_running("new instruction")
    thread.join(1)
    assert result == ["ack: new instruction"]
    assert "new instruction" in [message["content"] for message in agent.state.messages if message["role"] == "user" and isinstance(message["content"], str)]
    assert seen == ["new instruction"]
