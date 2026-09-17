"""Tools that let an agent maintain a focused, explicit task plan."""

from __future__ import annotations

from typing import Any

from kinetic_sdk.todo.models import TODO_STATUSES, TodoItem, TodoList
from kinetic_sdk.todo.store import TodoStore
from kinetic_sdk.tool.base import Tool, ToolResult

_STATUS_MARKERS = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def _render(todo_list: TodoList) -> str:
    """Render todos as a compact checklist suitable for an LLM tool result."""
    return "\n".join(
        f"{_STATUS_MARKERS[item.status]} {item.id}. {item.content}"
        for item in todo_list.items
    )


class TodoWriteTool(Tool):
    """Replace the complete todo scratchpad used for a multi-step task."""

    name = "todo_write"
    description = (
        "Maintain an explicit plan for a multi-step, non-trivial task; do not use it "
        "for a single trivial request. Send the full ordered todo list on every call. "
        "A newly created plan may contain only pending items; once work has started, "
        "keep exactly one unfinished item in_progress, mark an item completed "
        "immediately when it is finished (do not batch updates at the end), and then "
        "advance the next item. New-plan example: [{\"content\": \"Inspect tests\", "
        "\"status\": \"pending\"}, {\"content\": \"Implement fix\", "
        "\"status\": \"pending\"}]. Active-plan example: [{\"content\": \"Inspect tests\", "
        "\"status\": \"completed\"}, {\"content\": \"Implement fix\", "
        "\"status\": \"in_progress\"}, {\"content\": \"Run tests\", "
        "\"status\": \"pending\"}]."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Full ordered list; use [] only to clear completed work.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Existing id to preserve when round-tripping; omit for a position-based id.",
                        },
                        "content": {"type": "string", "description": "Short imperative task."},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, store: TodoStore) -> None:
        self._store = store

    def execute(self, todos: list[dict[str, Any]], **_: Any) -> ToolResult:  # type: ignore[override]
        """Replace the entire todo list with the provided items in one call."""
        if not isinstance(todos, list):
            return ToolResult(error="todos must be a list")

        items: list[TodoItem] = []
        for position, raw_item in enumerate(todos, start=1):
            if not isinstance(raw_item, dict):
                return ToolResult(error=f"todo at position {position} must be an object")
            content = raw_item.get("content")
            status = raw_item.get("status")
            item_id = raw_item.get("id", str(position))
            if not isinstance(content, str) or not content.strip():
                return ToolResult(error=f"todo at position {position} content must be a non-empty string")
            if not isinstance(status, str) or status not in TODO_STATUSES:
                return ToolResult(
                    error=f"todo at position {position} status must be one of {sorted(TODO_STATUSES)}"
                )
            if not isinstance(item_id, str) or not item_id.strip():
                return ToolResult(error=f"todo at position {position} id must be a non-empty string")
            items.append(TodoItem(item_id, content, status))

        if len({item.id for item in items}) != len(items):
            return ToolResult(error="todo ids must be unique")
        in_progress_count = sum(item.status == "in_progress" for item in items)
        all_completed = bool(items) and all(item.status == "completed" for item in items)
        all_pending = bool(items) and all(item.status == "pending" for item in items)
        if items and not all_completed and not all_pending and in_progress_count != 1:
            return ToolResult(
                error="exactly one todo must be in_progress unless the list is empty, all completed, or all pending"
            )

        todo_list = TodoList(items)
        try:
            self._store.save(todo_list)
        except Exception as exc:  # noqa: BLE001 - convert recoverable store failures for the model
            return ToolResult(error=f"could not save todos: {exc}")
        return ToolResult(output=_render(todo_list))


class TodoReadTool(Tool):
    """Read the current todo scratchpad without modifying it."""

    name = "todo_read"
    description = (
        "Read the current plan for a multi-step, non-trivial task before updating it. "
        "Use todo_write to change the full list, keeping exactly one unfinished item "
        "in_progress and completing items immediately. Example output: [~] 2. "
        "Implement fix. Do not use this tool for a single trivial request."
    )
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: TodoStore) -> None:
        self._store = store

    def execute(self, **_: Any) -> ToolResult:
        """Return the current todo list in the same rendered format."""
        try:
            todo_list = self._store.load()
        except Exception as exc:  # noqa: BLE001 - convert recoverable store failures for the model
            return ToolResult(error=f"could not load todos: {exc}")
        if todo_list is None or not todo_list.items:
            return ToolResult(output="No todos yet.")
        return ToolResult(output=_render(todo_list))
