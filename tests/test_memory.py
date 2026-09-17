"""Memory providers, explicit MemoryTool, and Agent memory wiring."""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.memory import (
    InMemoryMemory,
    JsonFileMemory,
    MemoryError,
    MemoryForgetTool,
    MemoryInspectTool,
    MemoryProvider,
    MemorySource,
    MemoryTier,
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
    assert json.loads(path.read_text())["schema_version"] == 2
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


def test_memory_tiers_provenance_and_scope_filtering():
    mem = InMemoryMemory()
    working = mem.add("docker working", {"workspace_id": "one", "user_id": "a"}, tier=MemoryTier.WORKING, source=MemorySource.TOOL_RESULT)
    persistent = mem.add("docker persistent", {"workspace_id": "one", "project_id": "p"}, source=MemorySource.USER_INPUT)
    mem.add("docker elsewhere", {"workspace_id": "two"}, source=MemorySource.LLM_INFERENCE)
    assert working.tier is MemoryTier.WORKING and working.source is MemorySource.TOOL_RESULT
    assert mem.search("docker", tiers={MemoryTier.PERSISTENT}, scope={"workspace_id": "one"}) == [persistent]
    assert mem.all(scope={"workspace_id": "one", "project_id": "p"}) == [persistent]


def test_memory_ttl_is_hidden_and_purge_removes_it():
    mem = InMemoryMemory()
    expired = mem.add("short lived", source=MemorySource.USER_INPUT, ttl_seconds=0)
    retained = mem.add("forever", source=MemorySource.USER_INPUT)
    assert mem.all() == [retained]
    assert mem.all(include_expired=True) == [expired, retained]
    assert mem.purge_expired() == 1
    assert mem.all(include_expired=True) == [retained]


def test_jsonfile_migrates_v1_entries_with_unknown_provenance(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": [{"id": "old", "text": "legacy", "metadata": {}, "created_at": "2025-01-01T00:00:00+00:00"}]}))
    entry = JsonFileMemory(path).all()[0]
    assert entry.tier is MemoryTier.PERSISTENT and entry.source is MemorySource.UNKNOWN


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


def test_memory_inspect_and_forget_tools_are_scoped():
    memory = InMemoryMemory()
    entry = memory.add("working fact", {"workspace_id": "one"}, tier=MemoryTier.WORKING, source=MemorySource.TOOL_RESULT)
    persistent = memory.add("persistent fact", {"workspace_id": "one"}, source=MemorySource.USER_INPUT)
    inspected = MemoryInspectTool(memory).execute(tiers=["working"], scope={"workspace_id": "one"})
    assert inspected.output["memories"][0]["id"] == entry.id
    forget = MemoryForgetTool(memory)
    assert forget.execute(scope={"workspace_id": "one"}, tiers=["working"]).output["deleted"] == 1
    assert forget.execute(scope={"workspace_id": "one"}, tiers=["persistent"]).is_error
    assert forget.execute(id=persistent.id).output["deleted"] is True


def test_agent_clears_working_memory_for_its_run():
    memory = InMemoryMemory()

    class AddWorkingMemory(MockLLMClient):
        def chat(self, messages, tools=None, system=None):
            memory.add("temporary", {"run_id": agent.run_id}, tier=MemoryTier.WORKING, source=MemorySource.LLM_INFERENCE)
            return text_response("done")

    agent = Agent(llm=AddWorkingMemory([]), memory=memory, permission_policy=PermissivePolicy())
    agent.run("finish")
    assert memory.all(tiers={MemoryTier.WORKING}, include_expired=True) == []


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
    assert stored.tier is MemoryTier.EPISODIC
    assert stored.source is MemorySource.LLM_INFERENCE


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
