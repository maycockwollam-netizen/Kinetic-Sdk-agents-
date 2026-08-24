"""``MemoryTool``: agent-driven long-term memory.

The agent itself decides what is worth remembering and what to look up —
the tool exposes the bound :class:`~kinetic_sdk.memory.provider.MemoryProvider`
through four actions (``store`` / ``search`` / ``list`` / ``clear``).
Automatic recall/store is a separate, narrower channel (``Agent(memory=...)``
recalls before a run and stores the final Q/A after it); this tool exists so
the agent can also manage memory explicitly mid-run.

Search results are returned as ``[{id, text, metadata}]`` dicts; the model
gets self-contained content it can cite without a second lookup.
"""

from __future__ import annotations

from typing import Any

from kinetic_sdk.memory.provider import MemoryEntry, MemoryProvider
from kinetic_sdk.tool.base import Tool, ToolResult


class MemoryTool(Tool):
    """Explicit memory access for the agent (store/search/list/clear).

    Args:
        provider: The memory backend to operate on.
        default_limit: Fallback cap for ``search`` when the model omits
            ``limit`` (hard-capped at :attr:`MAX_LIMIT`).
    """

    MAX_LIMIT = 20

    name = "memory"
    description = (
        "Long-term memory. Actions: 'store' (save a fact/lesson worth "
        "recalling in future sessions), 'search' (find memories relevant to "
        "a query), 'list' (show all memories), 'clear' (delete all)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "search", "list", "clear"],
                "description": "The memory operation to run.",
            },
            "text": {
                "type": "string",
                "description": "Content to store (required for 'store').",
            },
            "query": {
                "type": "string",
                "description": "Lookup text (required for 'search').",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for 'search' (capped at 20).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, provider: MemoryProvider, default_limit: int = 5) -> None:
        self.provider = provider
        self.default_limit = default_limit

    def execute(  # type: ignore[override]
        self,
        action: str,
        text: str = "",
        query: str = "",
        limit: int | None = None,
        **_: Any,
    ) -> ToolResult:
        if action == "store":
            if not text.strip():
                return ToolResult(error="'store' requires a non-empty 'text'")
            entry = self.provider.add(text)
            return ToolResult(output={"id": entry.id, "stored": True})
        if action == "search":
            if not query.strip():
                return ToolResult(error="'search' requires a non-empty 'query'")
            cap = limit if isinstance(limit, int) and limit > 0 else self.default_limit
            cap = min(cap, self.MAX_LIMIT)
            results = self.provider.search(query, cap)
            return ToolResult(
                output={
                    "memories": [self._entry_dict(e) for e in results],
                    "count": len(results),
                }
            )
        if action == "list":
            entries = self.provider.all()
            return ToolResult(
                output={
                    "memories": [self._entry_dict(e) for e in entries],
                    "count": len(entries),
                }
            )
        if action == "clear":
            self.provider.clear()
            return ToolResult(output={"cleared": True})
        return ToolResult(error=f"unknown action: {action!r}")

    @staticmethod
    def _entry_dict(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "text": entry.text,
            "metadata": entry.metadata,
            "created_at": entry.created_at,
        }
