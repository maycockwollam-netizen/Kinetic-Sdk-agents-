from __future__ import annotations

import pytest

from kinetic_sdk.memory import EmbeddingClient, MemoryTier, VectorMemory


class FakeEmbeddings(EmbeddingClient):
    def embed(self, texts: list[str]) -> list[list[float]]:
        table = {"cats": [1, 0], "dogs": [0, 1], "cat query": [1, 0]}
        return [table[text] for text in texts]


class TieEmbeddings(EmbeddingClient):
    def embed(self, texts: list[str]) -> list[list[float]]:
        table = {"a": [1, 0], "b": [1, 0], "c": [0, 1], "query": [1, 0]}
        return [table[text] for text in texts]


def test_vector_memory_ranks_cosine_similarity() -> None:
    memory = VectorMemory(FakeEmbeddings())
    cat, dog = memory.add("cats"), memory.add("dogs")
    assert memory.search("cat query") == [cat]
    assert dog in memory.all()


def test_vector_memory_embedding_failures_are_soft() -> None:
    class Broken(EmbeddingClient):
        def embed(self, texts: list[str]) -> list[list[float]]: raise RuntimeError("offline")
    memory = VectorMemory(Broken())
    stored = memory.add("still stored")
    assert memory.all() == [stored]
    assert memory.search("anything") == []


def test_vector_memory_rust_engine_matches_fallback(monkeypatch) -> None:
    pytest.importorskip("kinetic_sdk._kinetic_rs")
    from kinetic_sdk.memory import vector as vector_module

    memory = VectorMemory(TieEmbeddings())
    first = memory.add("a")
    memory.add("c")
    second = memory.add("b")  # same vector as "a": ties must break by index

    with_rust = [entry.id for entry in memory.search("query")]
    monkeypatch.setattr(vector_module, "_kinetic_rs", None)
    with_python = [entry.id for entry in memory.search("query")]

    # Both engines rank the tied [1,0] vectors with the higher index first.
    assert with_rust == with_python == [second.id, first.id]


def test_vector_memory_rust_respects_tier_filter() -> None:
    pytest.importorskip("kinetic_sdk._kinetic_rs")
    memory = VectorMemory(TieEmbeddings())
    working = memory.add("a", tier=MemoryTier.WORKING)
    persistent = memory.add("b", tier=MemoryTier.PERSISTENT)

    assert [entry.id for entry in memory.search("query", tiers={MemoryTier.PERSISTENT})] == [persistent.id]
    assert [entry.id for entry in memory.search("query", tiers={MemoryTier.WORKING})] == [working.id]
