"""Hook points into the agent lifecycle (Stage 3).

Hooks let SDK users intercept the agent loop at well-defined points without
subclassing :class:`~kinetic_sdk.agent.agent.Agent`: log extra telemetry,
mutate a tool input right before execution, or — the motivating use case —
implement a real confirmation UX by answering ``ON_PERMISSION_CHECK`` when
the permission policy flags a call with ``requires_confirmation=True``.

A hook is any callable ``(HookContext) -> HookResult | None``. It is
registered for one or more :class:`HookPoint` on a
:class:`~kinetic_sdk.hooks.registry.HookRegistry`, and the registry is handed
to the ``Agent`` (``hooks=...``). Hooks are fully opt-in: when no registry is
configured the agent loop pays zero overhead.

Safety rule (same as the rest of the SDK): a hook that raises never crashes
the agent — the registry catches the exception, emits ``hooks.error`` on the
event bus, and runs the remaining hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from kinetic_sdk.llm.client import LLMResponse
from kinetic_sdk.security.policy import PermissionDecision
from kinetic_sdk.tool.base import ToolResult


class HookPoint(Enum):
    """A point in the agent lifecycle where hooks may intercept."""

    #: Start of :meth:`Agent.run`, before classification. Context:
    #: ``user_message``, ``run_id``.
    BEFORE_RUN = "before_run"

    #: End of :meth:`Agent.run`, before ``agent.run_finished`` is emitted.
    #: Context: ``final_text``, ``run_id``.
    AFTER_RUN = "after_run"

    #: Right before each LLM call. Context: ``iteration``, ``run_id``.
    BEFORE_LLM_CALL = "before_llm_call"

    #: Right after each LLM call. Context: ``iteration``, ``llm_response``,
    #: ``run_id``.
    AFTER_LLM_CALL = "after_llm_call"

    #: Before a tool call is permission-checked and executed. Context:
    #: ``tool_name``, ``tool_input``, ``run_id``. A hook may return
    #: ``HookResult(should_continue=False)`` to cancel the call, or
    #: ``HookResult(modified_context={"tool_input": {...}})`` to replace the
    #: input that will be checked and executed.
    BEFORE_TOOL_CALL = "before_tool_call"

    #: After a tool actually executed (not on denial/cancellation). Context:
    #: ``tool_name``, ``tool_input``, ``tool_result``, ``run_id``.
    AFTER_TOOL_CALL = "after_tool_call"

    #: Fired when the permission decision flags ``requires_confirmation=True``.
    #: Context: ``tool_name``, ``tool_input``, ``permission_decision``,
    #: ``run_id``. Any hook returning ``HookResult(should_continue=True)``
    #: counts as an explicit confirmation and lets the call proceed; this is
    #: the extension point for a real confirmation UX (CLI prompt, UI popup,
    #: webhook, ...).
    ON_PERMISSION_CHECK = "on_permission_check"

    #: ``Agent.run`` is about to re-raise an unexpected exception. Context:
    #: ``error``, ``run_id``.
    ON_ERROR = "on_error"


@dataclass
class HookContext:
    """Everything a hook needs to know about the point it was triggered at.

    One shared dataclass with optional fields — which fields are populated
    depends on :attr:`point` (documented per :class:`HookPoint` member).
    """

    point: HookPoint
    run_id: str | None = None
    user_message: str | None = None
    final_text: str | None = None
    iteration: int | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: ToolResult | None = None
    permission_decision: PermissionDecision | None = None
    llm_response: LLMResponse | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """What a hook returns to influence the flow.

    Attributes:
        should_continue: ``True`` (default) lets the flow proceed. ``False``
            stops it: at ``BEFORE_TOOL_CALL`` the call is cancelled, at
            ``ON_PERMISSION_CHECK`` the call stays denied. At other points the
            value is collected but does not change control flow.
        modified_context: Optional light mutation. Currently only
            ``{"tool_input": {...}}`` at ``BEFORE_TOOL_CALL`` is honoured —
            the replacement input is what gets permission-checked, audited
            and executed. Hooks are not required to use this field.
    """

    should_continue: bool = True
    modified_context: dict[str, Any] | None = None


@runtime_checkable
class Hook(Protocol):
    """A callable intercepting one agent lifecycle point.

    Any function ``(HookContext) -> HookResult | None`` satisfies this
    protocol — returning ``None`` means "just observing, no opinion".
    """

    def __call__(self, context: HookContext) -> HookResult | None: ...
