"""Simple workflow builders composed from repeated :meth:`Agent.run` calls."""

from __future__ import annotations

from collections.abc import Callable

from kinetic_sdk.agent.agent import Agent


def goal_completion_loop(
    agent: Agent,
    goal: str,
    max_rounds: int = 10,
    is_done: Callable[[str], bool] | None = None,
) -> str:
    """Run until completion predicate succeeds or the round cap is reached."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    done = is_done if is_done is not None else lambda output: bool(output.strip())
    previous = ""
    for round_number in range(max_rounds):
        message = (
            goal
            if round_number == 0
            else f"Goal: {goal}\n\nPrevious result:\n{previous}\n\nContinue toward the goal."
        )
        previous = agent.run(message)
        if done(previous):
            break
    return previous


def iterative_refinement(
    agent: Agent, task: str, critic_prompt: str, max_rounds: int = 3
) -> str:
    """Run a task once, then repeatedly ask the same agent to improve it."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    output = agent.run(task)
    for _ in range(1, max_rounds):
        output = agent.run(f"{critic_prompt}\n\nPrevious output:\n{output}")
    return output
