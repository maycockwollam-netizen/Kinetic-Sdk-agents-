"""Run with ``python examples/workflow_goal_loop.py`` to execute a goal-completion loop."""

from kinetic_sdk.agent import Agent
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response
from kinetic_sdk.workflows import goal_completion_loop

if __name__ == "__main__":
    agent = Agent(MockLLMClient([text_response("goal complete")]), permission_policy=PermissivePolicy())
    print(goal_completion_loop(agent, "Finish the demonstration"))
