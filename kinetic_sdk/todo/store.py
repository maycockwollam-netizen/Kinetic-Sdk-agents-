"""Versioned, atomic persistence for agent-maintained todo lists."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from kinetic_sdk.todo.models import TodoList

TODO_SCHEMA_VERSION = 1


class TodoStoreError(Exception):
    """Raised when a todo list cannot be read or written safely."""


class TodoStore(ABC):
    """Persistence interface for one agent todo list."""

    @abstractmethod
    def save(self, todo_list: TodoList) -> None:
        """Atomically replace the saved list with *todo_list*."""

    @abstractmethod
    def load(self) -> TodoList | None:
        """Return the saved list, or ``None`` when no list exists yet."""

    @abstractmethod
    def delete(self) -> None:
        """Remove the saved list. No-op when it does not exist."""


class JsonFileTodoStore(TodoStore):
    """One todo JSON file, written via temp-file plus atomic rename."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, todo_list: TodoList) -> None:
        payload = {"schema_version": TODO_SCHEMA_VERSION, "todos": todo_list.to_dict()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self) -> TodoList | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TodoStoreError(f"Cannot load todos from {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or "todos" not in payload:
            raise TodoStoreError(f"Cannot load todos from {self.path}: not a todo file")
        if payload.get("schema_version") != TODO_SCHEMA_VERSION:
            raise TodoStoreError(
                f"Cannot load todos from {self.path}: schema version "
                f"{payload.get('schema_version')!r} is not supported "
                f"(expected {TODO_SCHEMA_VERSION})"
            )
        try:
            todos_data = payload["todos"]
            if not isinstance(todos_data, dict):
                raise ValueError("todos must be a dict")
            return TodoList.from_dict(todos_data)
        except (TypeError, ValueError) as exc:
            raise TodoStoreError(f"Cannot load todos from {self.path}: {exc}") from exc

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class InMemoryTodoStore(TodoStore):
    """Process-local todo store for tests, demos, and ephemeral agents."""

    def __init__(self) -> None:
        self._todo_list: TodoList | None = None

    def save(self, todo_list: TodoList) -> None:
        """Store a copy so caller mutations cannot alter persisted state."""
        self._todo_list = TodoList.from_dict(todo_list.to_dict())

    def load(self) -> TodoList | None:
        """Return a copy of the saved list, if any."""
        if self._todo_list is None:
            return None
        return TodoList.from_dict(self._todo_list.to_dict())

    def delete(self) -> None:
        """Clear the process-local list."""
        self._todo_list = None
