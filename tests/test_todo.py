"""Tests for the agent-maintained todo scratchpad tools and stores."""

from __future__ import annotations

import json

import pytest

from kinetic_sdk.todo import (
    InMemoryTodoStore,
    JsonFileTodoStore,
    TodoItem,
    TodoList,
    TodoReadTool,
    TodoStoreError,
    TodoWriteTool,
)


def _active_todos() -> list[dict[str, str]]:
    return [
        {"content": "Inspect repository", "status": "completed"},
        {"content": "Implement todo tool", "status": "in_progress"},
        {"content": "Run tests", "status": "pending"},
    ]


def test_todo_models_round_trip_and_reject_malformed_data():
    todo_list = TodoList([TodoItem("1", "Implement feature", "in_progress")])

    assert TodoList.from_dict(todo_list.to_dict()) == todo_list
    assert TodoItem.from_dict(TodoItem("1", "Implement feature", "pending").to_dict()) == TodoItem(
        "1", "Implement feature", "pending"
    )

    with pytest.raises(ValueError, match="status"):
        TodoItem.from_dict({"id": "1", "content": "Bad", "status": "blocked"})
    with pytest.raises(ValueError, match="content"):
        TodoItem.from_dict({"id": "1", "status": "pending"})
    with pytest.raises(ValueError, match="items"):
        TodoList.from_dict({})


def test_json_todo_store_round_trip_and_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "todos.json"
    store = JsonFileTodoStore(path)
    first = TodoList([TodoItem("1", "First task", "in_progress")])
    store.save(first)

    def fail_replace(_source, _destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("kinetic_sdk.todo.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.save(TodoList([TodoItem("1", "Replacement", "in_progress")]))

    assert store.load() == first
    assert list(tmp_path.glob("todos.json.*.tmp")) == []


def test_json_todo_store_rejects_corrupt_json_and_bad_schema(tmp_path):
    path = tmp_path / "todos.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(TodoStoreError, match="Cannot load todos"):
        JsonFileTodoStore(path).load()

    path.write_text(json.dumps({"schema_version": 99, "todos": {"items": []}}), encoding="utf-8")
    with pytest.raises(TodoStoreError, match="schema version"):
        JsonFileTodoStore(path).load()


def test_in_memory_todo_store_instances_are_independent_and_delete():
    first = InMemoryTodoStore()
    second = InMemoryTodoStore()
    todo_list = TodoList([TodoItem("1", "One", "in_progress")])

    first.save(todo_list)
    assert first.load() == todo_list
    assert second.load() is None
    first.delete()
    assert first.load() is None


def test_todo_write_accepts_valid_list_and_renders_checklist():
    store = InMemoryTodoStore()
    result = TodoWriteTool(store).execute(_active_todos())

    assert not result.is_error
    assert result.output == (
        "[x] 1. Inspect repository\n[~] 2. Implement todo tool\n[ ] 3. Run tests"
    )
    saved = store.load()
    assert saved is not None
    assert [item.id for item in saved.items] == ["1", "2", "3"]


def test_todo_write_rejects_zero_or_multiple_active_items_without_persisting():
    store = InMemoryTodoStore()
    tool = TodoWriteTool(store)
    assert not tool.execute(_active_todos()).is_error

    no_active = tool.execute(
        [
            {"content": "Done", "status": "completed"},
            {"content": "Still pending", "status": "pending"},
        ]
    )
    multiple_active = tool.execute(
        [
            {"content": "One", "status": "in_progress"},
            {"content": "Two", "status": "in_progress"},
        ]
    )

    assert no_active.is_error and "exactly one" in no_active.error
    assert multiple_active.is_error and "exactly one" in multiple_active.error
    assert TodoReadTool(store).execute().output == (
        "[x] 1. Inspect repository\n[~] 2. Implement todo tool\n[ ] 3. Run tests"
    )


def test_todo_write_accepts_all_completed_and_empty_list_to_clear():
    store = InMemoryTodoStore()
    tool = TodoWriteTool(store)

    completed = tool.execute([{"content": "Finished", "status": "completed"}])
    cleared = tool.execute([])

    assert completed.output == "[x] 1. Finished"
    assert cleared.output == ""
    assert TodoReadTool(store).execute().output == "No todos yet."


def test_todo_tools_preserve_round_tripped_ids_and_share_store():
    store = InMemoryTodoStore()
    write = TodoWriteTool(store)
    result = write.execute(
        [
            {"id": "auth", "content": "Fix authentication", "status": "in_progress"},
            {"id": "tests", "content": "Add regression tests", "status": "pending"},
        ]
    )

    assert result.output == "[~] auth. Fix authentication\n[ ] tests. Add regression tests"
    assert TodoReadTool(store).execute().output == result.output


def test_todo_read_returns_empty_state_before_any_write():
    result = TodoReadTool(InMemoryTodoStore()).execute()

    assert result.output == "No todos yet."
    assert not result.is_error
