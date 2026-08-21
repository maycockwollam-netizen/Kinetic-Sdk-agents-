"""Permission policies gating tool execution (Stage 3, part 1).

A policy decides — *before* a tool runs — whether the call may proceed. The
agent loop asks the configured :class:`PermissionPolicy` for every tool call
the model requests and never executes a call the policy rejects. This is the
SDK's main line of defence against blindly trusting LLM output when real
tools (terminal, filesystem, git, ...) are wired in.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PermissionDecision:
    """The verdict of a :class:`PermissionPolicy` for one tool call.

    Attributes:
        allowed: Whether the call may execute.
        reason: Human-readable explanation, surfaced to the model on denial
            and recorded in the audit log either way.
        requires_confirmation: When ``True`` the call is only allowed after
            explicit human confirmation (e.g. deleting files, ``rm -rf``,
            pushing to a remote). The agent loop currently denies such calls
            in automated mode; the flag exists so a future confirmation
            mechanism can let them through.
    """

    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False


class PermissionPolicy(ABC):
    """Interface for tool-call permission policies."""

    @abstractmethod
    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        """Decide whether *tool_name* may run with *tool_input*.

        Must be pure (no side effects) and fast: it runs before every single
        tool call in the agent loop.
        """


def _serialise_input(tool_input: dict[str, Any]) -> str:
    """Flatten a tool input to text so patterns can match across all values."""
    try:
        return json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(tool_input)


def _first_match(patterns: list[str], text: str) -> str | None:
    """Return the first pattern matching *text*, or ``None``.

    Patterns are treated as regular expressions; a pattern that fails to
    compile falls back to plain substring matching so simple strings like
    ``"rm -rf"`` work without escaping.
    """
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return pattern
        except re.error:
            if pattern in text:
                return pattern
    return None


class AllowListPolicy(PermissionPolicy):
    """Deny-by-default policy driven by an explicit allow-list.

    * Tools in ``always_allow`` may run freely...
    * ...unless their input matches one of the tool's
      ``require_confirmation_patterns`` (regex or substring), in which case
      the decision flags ``requires_confirmation`` (e.g. a ``terminal`` input
      containing ``rm -rf`` or ``sudo``, a ``git`` input containing ``push``).
    * Any tool not in the allow-list is denied outright
      (``allowed=False, reason="tool not in allow-list"``) — safe by default.
    """

    def __init__(
        self,
        always_allow: list[str] | None = None,
        require_confirmation_patterns: dict[str, list[str]] | None = None,
    ) -> None:
        self.always_allow: frozenset[str] = frozenset(always_allow or [])
        self.require_confirmation_patterns: dict[str, list[str]] = {
            name: list(patterns)
            for name, patterns in (require_confirmation_patterns or {}).items()
        }

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        if tool_name not in self.always_allow:
            return PermissionDecision(allowed=False, reason="tool not in allow-list")
        matched = _first_match(
            self.require_confirmation_patterns.get(tool_name, []),
            _serialise_input(tool_input),
        )
        if matched is not None:
            return PermissionDecision(
                allowed=True,
                reason=f"input matched dangerous pattern {matched!r}",
                requires_confirmation=True,
            )
        return PermissionDecision(allowed=True, reason="tool in allow-list")


class PermissivePolicy(PermissionPolicy):
    """Allow-everything policy for local dev/test environments only.

    ⚠️ KHÔNG dùng trong production — policy này bỏ qua mọi kiểm soát quyền
    hạn, agent có thể chạy bất kỳ tool nào (bao gồm lệnh huỷ diệt) mà không
    cần xác nhận. Chỉ dùng khi chạy local, trong sandbox, hoặc trong test.
    """

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        return PermissionDecision(allowed=True, reason="permissive policy (dev/test only)")
