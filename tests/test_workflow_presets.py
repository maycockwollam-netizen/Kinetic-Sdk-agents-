from kinetic_sdk.agent import Agent
from kinetic_sdk.testing import MockLLMClient, text_response
from kinetic_sdk.workflows import goal_completion_loop, iterative_refinement


def test_goal_completion_stops_when_predicate_succeeds():
    llm = MockLLMClient([text_response("not yet"), text_response("DONE")])
    assert goal_completion_loop(Agent(llm), "goal", max_rounds=5, is_done=lambda text: text == "DONE") == "DONE"
    assert len(llm.calls) == 2


def test_goal_completion_stops_at_round_limit():
    llm = MockLLMClient([text_response("a"), text_response("b")])
    assert goal_completion_loop(Agent(llm), "goal", max_rounds=2, is_done=lambda _: False) == "b"
    assert len(llm.calls) == 2


def test_iterative_refinement_passes_previous_output_to_each_round():
    llm = MockLLMClient([text_response("draft"), text_response("better"), text_response("best")])
    assert iterative_refinement(Agent(llm), "write", "Critique and improve", max_rounds=3) == "best"
    assert "draft" in llm.calls[1]["messages"][-1]["content"]
    assert "better" in llm.calls[2]["messages"][-1]["content"]
