"""The Kinetic agent: a tool-calling loop.

This module wires together the Stage 1 building blocks into the core agent
loop:

1. Read the conversation history from :class:`ConversationState`.
2. Ask the :class:`LLMClient` for the next turn (with the available tools).
3. If the model requested tool calls, execute each registered :class:`Tool`
   and append the results back to the conversation, then loop.
4. If the model produced a final text answer, return it.

Stage 1 implements the loop only. FLASH/MAX routing (``classifier.py``) and
context truncation (``context/manager.py``) are stubbed but not wired into the
loop yet - see those modules for the planned integration points. The loop is
synchronous and deterministic, which keeps tests simple.

The loop emits events on an optional :class:`EventBus` so observers can react
to each step without coupling to the agent internals.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.event.bus import EventBus, Event
from kinetic_sdk.llm.client import LLMClient, LLMResponse, ToolCall
from kinetic_sdk.tool.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class Agent:
    """A tool-calling agent bound to one LLM and a set of tools.

    Args:
        llm: The model client used for every reasoning turn.
        tools: Tools the agent may call. Duplicated tool names raise on
            construction to keep the dispatch table unambiguous.
        state: Conversation state. A fresh one is created if omitted.
        event_bus: Optional event bus the loop publishes lifecycle events to.
            Events emitted (see module docstring for the full list):
              ``agent.run_started``, ``agent.turn_started``,
              ``agent.llm_response``, ``agent.tool_call_started``,
              ``agent.tool_call_finished``, ``agent.run_finished``,
              ``agent.escalated``, ``agent.error``.
        max_iterations: Safety cap on LLM turns per :meth:`run` to prevent
            infinite tool-calling loops. The default is conservative.

    Attributes:
        mode: Current :class:`AgentMode`. Defaults to :attr:`AgentMode.MAX`
            in Stage 1 (routing lands in Stage 2). Escalation FLASH -> MAX is
            allowed via :meth:`escalate`; the reverse is not.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Iterable[Tool] | None = None,
        state: ConversationState | None = None,
        event_bus: EventBus | None = None,
        max_iterations: int = 25,
    ) -> None:
        self.llm = llm
        # NOTE: use ``is not None`` rather than truthiness because
        # ConversationState defines __len__ (an empty state is falsy but is
        # still a perfectly valid state object the caller passed in).
        self.state = state if state is not None else ConversationState()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.max_iterations = max_iterations
        self.mode: AgentMode = AgentMode.MAX

        tool_list = list(tools or [])
        self._tools: dict[str, Tool] = {}
        for tool in tool_list:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    # --- public API ---------------------------------------------------

    def add_tool(self, tool: Tool) -> None:
        """Register an additional tool at runtime.

        Raises if a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return the tool definitions to send to the model."""
        return [t.to_schema() for t in self._tools.values()]

    def run(self, user_message: str | None = None) -> str:
        """Run the agent loop until the model stops calling tools.

        Args:
            user_message: Optional user turn to append before running. Pass
                ``None`` to continue an existing conversation (e.g. after a
                tool result injected externally - not used in Stage 1).

        Returns:
            The final assistant text. If the loop hit ``max_iterations``
            without a final answer, returns the last assistant text seen
            (possibly empty) and publishes an ``agent.error`` event.
        """
        if user_message is not None:
            self.state.add_user_message(user_message)

        self._emit("agent.run_started", {"mode": self.mode.value, "tools": list(self._tools)})

        final_text = ""
        try:
            for iteration in range(self.max_iterations):
                self._emit("agent.turn_started", {"iteration": iteration})
                response = self._call_llm()
                self.state.add_assistant(self._assistant_content(response))
                self._emit("agent.llm_response", {"tool_calls": len(response.tool_calls), "stop_reason": response.stop_reason})

                if not response.tool_calls:
                    final_text = response.content
                    break

                self._execute_tool_calls(response.tool_calls)
            else:
                logger.warning("Agent hit max_iterations=%d", self.max_iterations)
                self._emit(
                    "agent.error",
                    {"reason": "max_iterations", "iterations": self.max_iterations},
                )
        except Exception as exc:
            self._emit("agent.error", {"reason": "exception", "error": str(exc)})
            raise

        self._emit("agent.run_finished", {"final_text": final_text, "mode": self.mode.value})
        return final_text

    def escalate(self) -> bool:
        """Escalate from FLASH to MAX mid-task. Returns True if it escalated.

        Downgrading is intentionally not supported within one task.
        """
        target = AgentMode.escalates_to(self.mode)
        if target is None:
            return False
        previous = self.mode
        self.mode = target
        self._emit("agent.escalated", {"from": previous.value, "to": target.value})
        return True

    # --- internals ----------------------------------------------------

    def _call_llm(self) -> LLMResponse:
        """Ask the LLM for the next turn using the current history + tools."""
        system, messages = self.state.for_llm()
        tools = self.tool_schemas() or None
        return self.llm.chat(messages=messages, tools=tools, system=system)

    def _assistant_content(self, response: LLMResponse) -> Any:
        """Build the assistant message ``content`` to store in history.

        Reproduces Anthropic's content-block shape so the history can be
        replayed to the model verbatim: text blocks + tool_use blocks.
        """
        blocks: list[dict[str, Any]] = []
        if response.content:
            blocks.append({"type": "text", "text": response.content})
        for call in response.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        return blocks if blocks else response.content

    def _execute_tool_calls(self, calls: list[ToolCall]) -> None:
        """Execute each tool call in order and append results to the state."""
        for call in calls:
            self._emit("agent.tool_call_started", {"name": call.name, "id": call.id})
            result = self._execute_one(call)
            self._emit(
                "agent.tool_call_finished",
                {
                    "name": call.name,
                    "id": call.id,
                    "is_error": result.is_error,
                    "output_preview": self._preview(result.output),
                },
            )
            self.state.add_tool_result(
                call.id,
                self._format_tool_output(result),
                is_error=result.is_error,
            )

    def _execute_one(self, call: ToolCall) -> ToolResult:
        """Dispatch a single tool call, mapping failures to ToolResult errors."""
        tool = self._tools.get(call.name)
        if tool is None:
            logger.error("Unknown tool requested: %s", call.name)
            return ToolResult(error=f"Unknown tool: {call.name}")
        try:
            return tool.execute(**call.arguments)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            logger.exception("Tool %s raised", call.name)
            return ToolResult(error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _format_tool_output(result: ToolResult) -> str:
        """Serialise a ToolResult to a string the model can read back."""
        if result.is_error:
            return json.dumps({"error": result.error})
        payload = result.output
        try:
            return json.dumps(payload) if not isinstance(payload, str) else payload
        except (TypeError, ValueError):
            return str(payload)

    @staticmethod
    def _preview(value: Any, limit: int = 200) -> str:
        """Truncate a value to a short string for event payloads/logs."""
        s = str(value)
        return s if len(s) <= limit else s[:limit] + "..."

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish a lifecycle event if a bus is attached."""
        if self.event_bus is None:
            return
        self.event_bus.publish(Event(type=event_type, payload=payload, source="agent"))
