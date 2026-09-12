"""Long-term memory for agents: providers, search, and the explicit tool.

Minimal wiring example::

    from kinetic_sdk import Agent
    from kinetic_sdk.memory import InMemoryMemory, MemoryTool

    memory = InMemoryMemory()
    agent = Agent(
        llm=client,
        tools=[MemoryTool(memory)],
        memory=memory,  # auto-recall before runs + auto-store after them
        permission_policy=my_policy_allows_memory,
    )

The ``Agent(memory=...)`` integration is append-only convenience: it recalls
relevant entries before a run (injected as a separate user message) and
stores the Q/A pair after it. Explicit mid-run memory management goes
through :class:`~kinetic_sdk.memory.tool.MemoryTool` like any tool call.
Embedding/vector providers plug into the same ABC when keyword overlap is
not enough. ``VectorMemory`` has an injectable backend; its LiteLLM wrapper
is lazy-imported and remains optional.
"""

from kinetic_sdk.memory.inmemory import InMemoryMemory
from kinetic_sdk.memory.json_file import JsonFileMemory
from kinetic_sdk.memory.provider import (
    MemoryEntry,
    MemoryError,
    MemoryProvider,
    relevance,
    tokens,
)
from kinetic_sdk.memory.tool import MemoryTool
from kinetic_sdk.memory.vector import (
    EmbeddingClient,
    LiteLLMEmbeddingClient,
    VectorMemory,
)

__all__ = [
    "InMemoryMemory",
    "JsonFileMemory",
    "MemoryEntry",
    "MemoryError",
    "MemoryProvider",
    "MemoryTool",
    "EmbeddingClient",
    "LiteLLMEmbeddingClient",
    "VectorMemory",
    "relevance",
    "tokens",
]
