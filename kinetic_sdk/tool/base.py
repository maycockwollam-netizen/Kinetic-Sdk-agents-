"""Abstract tool interface for the Kinetic Agent SDK.

Every capability exposed to the agent (terminal, file editor, web search, ...)
is implemented as a :class:`Tool`. The agent loop interacts with tools purely
through this interface, so concrete implementations can be swapped without
touching the agent code.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolFailureCategory(str, Enum):
    """How the agent can safely recover from a failed tool invocation."""

    RECOVERABLE_INPUT = "recoverable_input"
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class ToolRiskLevel(str, Enum):
    """The side-effect risk a tool voluntarily declares to the agent."""

    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class ToolCapability(str, Enum):
    """A resource domain a tool voluntarily declares it can access."""

    LOCAL = "local"
    NETWORK = "network"
    FILESYSTEM = "filesystem"


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Optional per-tool execution overrides.

    ``None`` for ``timeout_seconds`` and ``max_transient_retries`` retains
    the agent-wide historical settings. ``None`` for
    ``circuit_breaker_threshold`` disables the circuit breaker for this tool,
    preserving historical behaviour.
    """

    timeout_seconds: float | None = None
    max_transient_retries: int | None = None
    circuit_breaker_threshold: int | None = None
    circuit_breaker_cooldown_seconds: float = 30.0


@dataclass
class ToolResult:
    """Structured result returned by a tool execution.

    Attributes:
        output: The primary textual/structured output produced by the tool.
            Keep it JSON-serialisable so it can be embedded back into the
            conversation history sent to the model.
        error: Optional error message. When non-empty, the agent treats the
            execution as failed and may retry or surface it to the user.
        metadata: Optional bag of extra information (timing, token counts,
            file paths touched, ...). Never used for control flow.
        failure_category: Optional recovery hint for failed results. ``None``
            preserves the historical behaviour: the result is returned to the
            model without an automatic retry.
        duration_seconds: Wall-clock duration measured by the agent around
            ``execute``. ``None`` means the result did not come from an agent
            execution path (the historical shape).
        artifacts: Optional paths or identifiers for supplementary data
            produced by the tool. An empty list preserves historical behaviour.
    """

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_category: ToolFailureCategory | None = None
    duration_seconds: float | None = None
    artifacts: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        """True when this result represents a failed execution."""
        return self.error is not None and self.error != ""


class Tool(ABC):
    """Abstract base class for all agent tools.

    Subclasses must declare a ``name``, ``description`` and ``parameters``
    JSON schema, and implement :meth:`execute`. The schema is sent to the
    model so it knows how to call the tool; ``execute`` runs the actual work.

    The class is intentionally lightweight (no decorators / registration
    magic) to keep the interface easy to test and replace.
    """

    #: Stable identifier sent to the model. Must be unique across a single
    #: agent's tool set. Declared as a plain (instance) annotation, NOT
    #: ``ClassVar``, so both declaration styles are valid: fixed tools set it
    #: as a class attribute (``name: ClassVar[str] = "git"``) while dynamic
    #: tools (``MockTool``, ``MCPToolAdapter``) assign it per instance.
    name: str

    #: Human-readable description shown to the model to help it decide when
    #: the tool is appropriate.
    description: str

    #: JSON Schema describing the parameters object the model should supply.
    #: Use ``type: object`` with ``properties`` for the fields you expect.
    parameters: dict[str, Any]

    #: Optional side-effect declaration. ``None`` keeps historical fail-safe
    #: behaviour: consumers that need a risk decision must treat the tool as
    #: write-capable.
    risk_level: ToolRiskLevel | None = None

    #: Optional resource-domain declarations. An empty set keeps historical
    #: behaviour and makes no assertion about the tool's capabilities.
    capabilities: frozenset[ToolCapability] = frozenset()

    #: Optional per-tool execution policy. ``None`` uses the agent-wide
    #: timeout/retry settings and has no circuit breaker, as before.
    execution_policy: ToolExecutionPolicy | None = None

    @abstractmethod
    def execute(self, *, idempotency_key: str | None = None, **params: Any) -> ToolResult:
        """Run the tool with validated parameters and return a result.

        Args:
            idempotency_key: Optional stable key supplied by the agent for a
                write/destructive call that may be retried. ``None`` preserves
                historical invocation behaviour; tools may ignore this value.
            **params: Keyword arguments matching :attr:`parameters`. The
                agent loop is responsible for extracting these from the
                model's tool call payload before invoking this method.

        Returns:
            A :class:`ToolResult`. Raise exceptions only for truly
            unexpected failures; recoverable errors should be returned via
            ``ToolResult(error=...)`` so the agent can react.
        """

    async def execute_async(self, **params: Any) -> ToolResult:
        """Async variant of :meth:`execute`, used by the async agent loop.

        The default implementation runs the synchronous :meth:`execute` in a
        worker thread (``asyncio.to_thread``) so existing sync tools work
        under :class:`~kinetic_sdk.agent.async_agent.AsyncAgent` without
        blocking the event loop. Tools with a natively async backend (async
        HTTP client, async subprocess, ...) should override this method —
        overriding ``execute`` is still required by the ABC, so a pure-async
        tool typically implements ``execute`` as a thin
        ``asyncio.run(self.execute_async(...))`` wrapper or raises
        ``NotImplementedError`` when it must never run synchronously.

        Note on cancellation: when the async loop abandons an ``await`` of
        this method (e.g. a tool timeout), a native-async override is
        cancelled for real, while the default thread bridge keeps running in
        the background — Python cannot safely kill a running thread.
        """
        return await asyncio.to_thread(self.execute, **params)

    def to_schema(self) -> dict[str, Any]:
        """Return the tool definition in a provider-neutral shape.

        The default format mirrors Anthropic's tool schema, which is also
        trivially convertible to OpenAI's function-calling format::

            {
                "name": ...,
                "description": ...,
                "input_schema": {...},
            }
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Tool {self.name}>"
