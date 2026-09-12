"""Run with ``python examples/file_based_subagent.py`` to load and spawn an agent.md subagent."""

import tempfile
from pathlib import Path

from kinetic_sdk.agent import Agent
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.subagent import SpawnBudget, load_subagent_spec, spawn_subagent
from kinetic_sdk.testing import MockLLMClient, text_response

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "agent.md"
        path.write_text("---\nname: reviewer\ndescription: Reviews code\n---\nReview the task carefully.", encoding="utf-8")
        parent = Agent(MockLLMClient([text_response("parent")]), permission_policy=PermissivePolicy())
        child = spawn_subagent(parent, load_subagent_spec(path), SpawnBudget(10))
        child.llm = MockLLMClient([text_response("review complete")])
        print(child.run("review this change"))
