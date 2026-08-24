"""Memory providers, explicit MemoryTool, and Agent memory wiring."""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.memory import (
    InMemoryMemory,
    JsonFileMemory,
    MemoryError,
    MemoryProvider,
    MemoryTool,
    relevance,
    tokens,
)
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response

# --- provider basics ----------------------------------------------------


def test_tokens_extract_content_words():
    t = tokens("The cat sat on the mat")
    assert "cat" in t and "mat" in t and "the" not in t


def test_relevance_scales():
    assert relevance("deploy pipeline", "deploy the pipeline config") == 1.0
    assert relevance("deploy pipeline", "unrelated text") == 0.0


def test_inmemory_add_and_search():
    mem = InMemoryMemory()
    mem.add("the deploy pipeline uses docker exec")
    mem.add("totally unrelated fact")
    hits = mem.search("docker pipeline")
    assert hits and "docker" in hits[0].text
    assert mem.search("docker pipeline", limit=1)[0].id == hits[0].id


def test_inmemory_empty_text_raises():
    mem = InMemoryMemory()
    with pytest.raises(MemoryError):
        mem.add("   ")


def test_inmemory_search_ignores_zero_scores():
    mem = InMemoryMemory()
    mem.add("unrelated words")
    assert mem.search("docker exec") == []


def test_inmemory_newest_wins_on_tie():
    mem = InMemoryMemory()
    a = mem.add("docker alpha")
    mem.add("docker beta")
    hits = mem.search("docker")
    assert hits[0].id != a.id  # newest (beta) first


def test_jsonfile_roundtrip(tmp_path):
    path = tmp_path / "memory.json"
    mem = JsonFileMemory(path)
    entry = mem.add("remember docker settings", metadata={"source": "test"})
    assert json.loads(path.read_text())["schema_version"] == 1
    reloaded = JsonFileMemory(path)
    assert reloaded.all()[0].id == entry.id
    assert reloaded.search("docker settings")


def test_jsonfile_clear_removes_file(tmp_path):
    path = tmp_path / "memory.json"
    mem = JsonFileMemory(path)
    mem.add("x")
    mem.clear()
    assert not path.exists()


def test_jsonfile_corrupt_raises(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ nope")
    with pytest.raises(MemoryError):
        JsonFileMemory(path)


def test_jsonfile_wrong_schema_raises(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema_version": 9, "entries": []}))
    with pytest.raises(MemoryError):
        JsonFileMemory(path)


# --- MemoryTool ---------------------------------------------------------


def _agent_with_memory_tool():
    memory = InMemoryMemory()
    tool = MemoryTool(memory)
    llm = MockLLMClient(
        [
            tool_response("c1", "memory", {"action": "store", "text": "fact"}),
            text_response("stored"),
        ]
    )
    agent = Agent(llm=llm, tools=[tool], permission_policy=PermissivePolicy())
    return agent, memory


def test_memory_tool_store_search_list_clear():
    memory = InMemoryMemory()
    tool = MemoryTool(memory)
    result = tool.execute(action="store", text="deploy uses docker")
    assert not result.is_error and result.output["stored"]
    result = tool.execute(action="search", query="docker")
    assert result.output["count"] == 1
    result = tool.execute(action="list")
    assert result.output["count"] == 1
    result = tool.execute(action="clear")
    assert result.output["cleared"] and memory.all() == []


def test_memory_tool_validates_missing_args():
    tool = MemoryTool(InMemoryMemory())
    assert tool.execute(action="store", text="  ").is_error
    assert tool.execute(action="search", query="").is_error
    assert tool.execute(action="bogus").is_error


def test_memory_tool_caps_limit():
    memory = InMemoryMemory()
    for i in range(30):
        memory.add(f"docker fact {i}")
    tool = MemoryTool(memory)
    result = tool.execute(action="search", query="docker", limit=100)
    assert result.output["count"] == tool.MAX_LIMIT


def test_agent_calls_memory_tool_end_to_end():
    agent, memory = _agent_with_memory_tool()
    agent.run("remember this")
    assert memory.all()[0].text == "fact"


# --- Agent wiring -------------------------------------------------------


def test_agent_auto_recall_and_store():
    memory = InMemoryMemory()
    memory.add("project uses docker exec for tests")
    llm = MockLLMClient([text_response("ok")])
    agent = Agent(
        llm=llm, memory=memory, permission_policy=PermissivePolicy()
    )
    agent.run("how do we run docker exec here?")
    # Recall message injected before the user's own message.
    assert any(
        m.get("role") == "user" and "[Recalled memories]" in str(m.get("content"))
        for m in agent.state.messages
    )
    # Q/A stored back.
    stored = memory.all()[-1]
    assert "docker exec" in stored.text and "ok" in stored.text


def test_agent_no_memory_no_injection():
    llm = MockLLMClient([text_response("ok")])
    agent = Agent(llm=llm, permission_policy=PermissivePolicy())
    agent.run("topic without recall")
    assert all(
        "[Recalled memories]" not in str(m.get("content"))
        for m in agent.state.messages
    )


def _BrokenMemory(MemoryProvider):
    class Broken(MemoryProvider):
        def add(self, text, metadata=None):
            raise RuntimeError("broken")

        def search(self, query, limit=5):
            raise RuntimeError("broken")

        def all(self):
            return []

        def clear(self) -> None:
            return None

    return Broken()


def test_agent_broken_memory_fail_soft():
    llm = MockLLMClient([text_response("ok")])
    agent = Agent(
        llm=llm, memory=_BrokenMemory(MemoryProvider), permission_policy=PermissivePolicy()
    )
    result = agent.run("hi")  # recall + store both raise internally; run survives
    assert result == "ok"
