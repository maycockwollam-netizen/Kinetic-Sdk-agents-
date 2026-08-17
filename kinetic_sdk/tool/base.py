"""Abstract tool interface for the Kinetic Agent SDK.

Every capability exposed to the agent (terminal, file editor, web search, ...)
is implemented as a :class:`Tool`. The agent loop interacts with tools purely
through this interface, so concrete implementations can be swapped without
touching the agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


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
    """

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
    #: agent's tool set.
    name: ClassVar[str]

    #: Human-readable description shown to the model to help it decide when
    #: the tool is appropriate.
    description: ClassVar[str]

    #: JSON Schema describing the parameters object the model should supply.
    #: Use ``type: object`` with ``properties`` for the fields you expect.
    parameters: ClassVar[dict[str, Any]]

    @abstractmethod
    def execute(self, **params: Any) -> ToolResult:
        """Run the tool with validated parameters and return a result.

        Args:
            **params: Keyword arguments matching :attr:`parameters`. The
                agent loop is responsible for extracting these from the
                model's tool call payload before invoking this method.

        Returns:
            A :class:`ToolResult`. Raise exceptions only for truly
            unexpected failures; recoverable errors should be returned via
            ``ToolResult(error=...)`` so the agent can react.
        """

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
