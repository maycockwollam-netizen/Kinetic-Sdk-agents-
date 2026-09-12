from __future__ import annotations

from kinetic_sdk.memory import EmbeddingClient, VectorMemory


class FakeEmbeddings(EmbeddingClient):
    def embed(self, texts: list[str]) -> list[list[float]]:
        table = {"cats": [1, 0], "dogs": [0, 1], "cat query": [1, 0]}
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
