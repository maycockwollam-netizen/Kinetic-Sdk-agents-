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


# --- persistence --------------------------------------------------------

def test_vector_memory_persists_entries_and_vectors(tmp_path) -> None:
    path = tmp_path / "mem.json"
    memory = VectorMemory(FakeEmbeddings(), path=path)
    cat = memory.add("cats")
    memory.add("dogs")

    reloaded = VectorMemory(FakeEmbeddings(), path=path)
    assert [entry.id for entry in reloaded.all()] == [entry.id for entry in memory.all()]
    assert [entry.text for entry in reloaded.all()] == ["cats", "dogs"]
    # Search against RELOADED vectors: the cosine ranking must survive a restart.
    assert reloaded.search("cat query") == [cat]
    assert path.exists()
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == VectorMemory.SCHEMA_VERSION


def test_vector_memory_clear_and_delete_persist(tmp_path) -> None:
    path = tmp_path / "mem.json"
    memory = VectorMemory(FakeEmbeddings(), path=path)
    cat = memory.add("cats")
    memory.add("dogs")

    memory.delete(cat.id)
    reloaded = VectorMemory(FakeEmbeddings(), path=path)
    assert [e.text for e in reloaded.all()] == ["dogs"]

    memory.clear()
    reloaded = VectorMemory(FakeEmbeddings(), path=path)
    assert reloaded.all() == []


def test_vector_memory_none_vector_persists_and_stays_unsearchable(tmp_path) -> None:
    class Broken(EmbeddingClient):
        def embed(self, texts: list[str]) -> list[list[float]]: raise RuntimeError("offline")

    path = tmp_path / "mem.json"
    memory = VectorMemory(Broken(), path=path)
    stored = memory.add("secret")

    reloaded = VectorMemory(Broken(), path=path)
    assert reloaded.all() == [stored]
    assert reloaded.search("secret") == []  # no vector -> excluded, fail-soft


def test_vector_memory_corrupt_or_wrong_schema_raises(tmp_path) -> None:
    from kinetic_sdk.memory import MemoryError

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(MemoryError, match="Cannot load"):
        VectorMemory(FakeEmbeddings(), path=bad_json)

    wrong_schema = tmp_path / "schema.json"
    wrong_schema.write_text('{"schema_version": 99, "entries": [], "vectors": []}', encoding="utf-8")
    with pytest.raises(MemoryError, match="schema version"):
        VectorMemory(FakeEmbeddings(), path=wrong_schema)

    mismatched = tmp_path / "mismatch.json"
    mismatched.write_text('{"schema_version": 1, "entries": [{}], "vectors": []}', encoding="utf-8")
    with pytest.raises(MemoryError, match="length mismatch"):
        VectorMemory(FakeEmbeddings(), path=mismatched)


def test_vector_memory_without_path_stays_in_memory() -> None:
    memory = VectorMemory(FakeEmbeddings())
    assert memory.path is None
    memory.add("cats")
    assert memory.search("cat query")  # still ranks


def test_vector_memory_preserves_lifecycle_fields_on_reload(tmp_path) -> None:
    from kinetic_sdk.memory import MemorySource, MemoryTier

    path = tmp_path / "lifecycle.json"
    memory = VectorMemory(FakeEmbeddings(), path=path)
    entry = memory.add(
        "fact",
        metadata={"k": "v"},
        tier=MemoryTier.EPISODIC,
        source=MemorySource.LLM_INFERENCE,
        ttl_seconds=3600,
    )

    reloaded = VectorMemory(FakeEmbeddings(), path=path)
    got = reloaded.all()[0]
    assert got.id == entry.id
    assert got.text == "fact"
    assert got.tier is MemoryTier.EPISODIC
    assert got.source is MemorySource.LLM_INFERENCE
    assert got.metadata == {"k": "v"}
    assert got.expires_at == entry.expires_at


def test_vector_memory_concurrent_persist_never_loses_entries(tmp_path, monkeypatch) -> None:
    """Two writers racing persist must not let an older snapshot clobber a newer one.

    Regression for the race-condition finding: ``_persist`` used to snapshot
    under the data lock, then release it before writing + ``os.replace``, so
    thread A could rename an OLDER snapshot last and drop thread B's entry
    from disk. The persist lock must cover snapshot -> serialize -> rename.

    The test pins the old interleaving: writer A's ``os.replace`` is blocked
    while writer B runs; B is given time to finish its whole write, then A is
    released so its OLDER snapshot lands on disk LAST. Under the pre-fix code
    that drops B; under the fix, B could not write at all while A held the
    persist lock, so A writes first, B then rewrites BOTH entries.
    """
    import os
    import threading
    import time

    path = tmp_path / "mem.json"
    first_replace_ready = threading.Event()
    release_first_replace = threading.Event()
    real_replace = os.replace

    def blocking_replace(src, dst):
        if os.fspath(dst) == os.fspath(path) and not first_replace_ready.is_set():
            first_replace_ready.set()
            assert release_first_replace.wait(5), "first writer never released"
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", blocking_replace)
    memory = VectorMemory(FakeEmbeddings(), path=path)

    first = threading.Thread(target=memory.add, args=("cats",))
    first.start()
    # Let the first persist reach its os.replace (snapshot [cats]) and block there.
    assert first_replace_ready.wait(5)
    second = threading.Thread(target=memory.add, args=("dogs",))
    second.start()
    # Under the FIXED code the second writer is blocked on the persist lock
    # during this wait, so giving it time to write is harmless; under the old
    # code it finishes writing [cats, dogs] before the first writer renames.
    time.sleep(0.1)
    release_first_replace.set()
    first.join()
    second.join()

    reloaded = VectorMemory(FakeEmbeddings(), path=path)
    assert {entry.text for entry in reloaded.all()} == {"cats", "dogs"}
