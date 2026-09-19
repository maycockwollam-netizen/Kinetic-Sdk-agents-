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
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse


@dataclass
class PermissionDecision:
    """The verdict of a :class:`PermissionPolicy` for one tool call.

    Attributes:
        allowed: Whether the call may execute.
        reason: Human-readable explanation, surfaced to the model on denial
            and recorded in the audit log either way.
        requires_confirmation: When ``True`` the call is only allowed after
            explicit human confirmation (e.g. deleting files, ``rm -rf``,
            pushing to a remote). The agent loop asks the
            ``ON_PERMISSION_CHECK`` hooks (see :mod:`kinetic_sdk.hooks`) to
            supply that confirmation; when no hook confirms, the call is
            denied — the safe default.
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


@dataclass
class PolicyRule:
    """One ordered, content-aware permission rule.

    Every populated selector must match for the rule to apply.  Tool inputs
    use the small cross-tool vocabulary ``path``, ``url``/``domain`` and
    ``command``.  A missing or non-string value simply makes its selector not
    match; policies must never make an otherwise safe tool call crash.
    """

    tool_name: str | None = None
    path_prefix: str | None = None
    domain_pattern: str | None = None
    command_pattern: str | None = None
    decision: PermissionDecision = field(
        default_factory=lambda: PermissionDecision(allowed=False)
    )


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


def _matches_pattern(pattern: str, text: str) -> bool:
    """Match a regex, falling back to literal containment if it is invalid."""
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return pattern in text


def _input_domain(tool_input: dict[str, Any]) -> str | None:
    """Return a normalised domain from the policy's supported input keys."""
    domain = tool_input.get("domain")
    if isinstance(domain, str) and domain:
        return domain
    url = tool_input.get("url")
    if not isinstance(url, str) or not url:
        return None
    # A bare domain is convenient for tools which accept either a URL or host.
    parsed = urlparse(url if "://" in url else f"//{url}")
    return parsed.hostname


class RuleBasedPolicy(PermissionPolicy):
    """Evaluate ordered rules against a tool's name and standardised input.

    The first matching rule wins; no match returns ``default_decision``.  This
    complements (rather than replaces) :class:`AllowListPolicy`: use it when
    a policy must distinguish e.g. a read-only terminal command from a
    destructive one, or requests to trusted domains from arbitrary hosts.
    """

    def __init__(
        self,
        rules: list[PolicyRule],
        default_decision: PermissionDecision | None = None,
    ) -> None:
        self.rules = list(rules)
        self.default_decision = (
            default_decision
            if default_decision is not None
            else PermissionDecision(allowed=False, reason="no policy rule matched")
        )

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionDecision:
        for rule in self.rules:
            if rule.tool_name is not None and rule.tool_name != tool_name:
                continue
            if rule.path_prefix is not None:
                path = tool_input.get("path")
                if not isinstance(path, str) or not path.startswith(rule.path_prefix):
                    continue
            if rule.domain_pattern is not None:
                domain = _input_domain(tool_input)
                if domain is None or not _matches_pattern(rule.domain_pattern, domain):
                    continue
            if rule.command_pattern is not None:
                command = tool_input.get("command")
                if not isinstance(command, str) or not _matches_pattern(
                    rule.command_pattern, command
                ):
                    continue
            # PermissionDecision is mutable for backwards compatibility;
            # return a copy so callers cannot mutate the reusable rule.
            return replace(rule.decision)
        return replace(self.default_decision)


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
