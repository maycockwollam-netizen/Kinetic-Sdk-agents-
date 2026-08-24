"""``AskUserTool``: the agent asks the operator a question mid-run.

Permission hooks answer "may I do that?"; this tool answers "I cannot
decide — give me input". That human-in-the-loop reversal (Claude SDK's
``AskUserQuestion``, AutoGen's ``UserProxyAgent``) is the difference between
an agent that halts on ambiguity and one that keeps working.

The answer channel is an injectable ``handler(question) -> str``, so CLI
apps, chat UIs and tests each plug their own; without one, the fallback is
a bare ``input()`` prompt. Like every tool, the call itself passes through
``permission_policy.check`` — policy decides whether the model may ask
questions at all.
"""

from __future__ import annotations

from typing import Any, Callable

from kinetic_sdk.tool.base import Tool, ToolResult


class AskUserTool(Tool):
    """Ask the human operator a question and return their answer.

    Args:
        handler: ``(question) -> answer`` callable. ``None`` falls back to
            a simple CLI prompt (``input``). Exceptions in the handler are
            returned as tool errors so the model can react.
    """

    name = "ask_user"
    description = (
        "Ask the human operator a question when you are blocked or need "
        "input. Returns the operator's answer verbatim. Use sparingly: "
        "every call waits on a human."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question posed to the operator.",
            },
        },
        "required": ["question"],
    }

    def __init__(self, handler: Callable[[str], str] | None = None) -> None:
        self._handler = handler

    def execute(self, question: str, **_: Any) -> ToolResult:  # type: ignore[override]
        if not isinstance(question, str) or not question.strip():
            return ToolResult(error="question must be a non-empty string")
        try:
            answer = (
                self._handler(question)
                if self._handler is not None
                else input(f"[ask_user] {question}\n> ")
            )
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            return ToolResult(error=f"ask_user handler failed: {exc}")
        return ToolResult(output=str(answer))
