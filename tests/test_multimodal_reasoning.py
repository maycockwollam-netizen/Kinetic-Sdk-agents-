from kinetic_sdk.agent import Agent
from kinetic_sdk.conversation import ConversationState
from kinetic_sdk.event import EventBus
from kinetic_sdk.llm import LiteLLMClient, LLMResponse
from kinetic_sdk.testing import MockLLMClient


def test_add_user_message_with_image_preserves_content_blocks():
    state = ConversationState()
    state.add_user_message("plain text")
    message = state.add_user_message_with_image("look", "YWJj")
    assert state.messages[0]["content"] == "plain text"
    assert message["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}},
    ]


def test_translate_messages_converts_image_block_to_openai_shape():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"},
                },
            ],
        }
    ]

    translated = LiteLLMClient._translate_messages(messages, system=None)

    assert translated[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
        ],
    }


def test_image_message_pipeline_translates_state_to_openai_shape():
    state = ConversationState()
    state.add_user_message_with_image("look", "YWJj")

    system, messages = state.for_llm()
    translated = LiteLLMClient._translate_messages(messages, system=system)

    assert translated == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
            ],
        }
    ]


def test_reasoning_text_is_best_effort():
    assert LLMResponse(raw={"reasoning": "careful analysis"}).reasoning_text == "careful analysis"
    assert LLMResponse(raw={"content": [{"type": "thinking", "thinking": "step one"}]}).reasoning_text == "step one"
    assert LLMResponse(raw={"id": "response"}).reasoning_text is None


def test_agent_emits_reasoning_trace_only_when_present():
    bus = EventBus()
    events = []
    bus.subscribe("agent.reasoning_trace", events.append)
    Agent(MockLLMClient([LLMResponse(content="done", raw={"reasoning": "why"})]), event_bus=bus).run("hi")
    assert events[0].payload["text"] == "why"
    events.clear()
    Agent(MockLLMClient([LLMResponse(content="done")]), event_bus=bus).run("hi")
    assert events == []
