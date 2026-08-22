"""Security package: permission policies, audit logging, secret redaction.

Confirmation UX
---------------
When a policy flags a call ``requires_confirmation=True``, the agent loop
consults the ``ON_PERMISSION_CHECK`` hooks (see :mod:`kinetic_sdk.hooks`)
before deciding: any hook returning ``HookResult(should_continue=True)``
confirms the call and lets it execute. With no hook answering (or none
configured) the call is denied, exactly as before hooks existed — the safe
default never changes implicitly.

The SDK core deliberately ships no concrete confirmation UI; the example
below shows how an SDK user wires a manual CLI prompt — the same hook shape
works for a UI popup, a Slack/webhook approval flow, or a rule-based
auto-approver::

    from kinetic_sdk.agent.agent import Agent
    from kinetic_sdk.hooks import HookContext, HookPoint, HookRegistry, HookResult

    def cli_confirm(context: HookContext) -> HookResult:
        answer = input(
            f"Tool {context.tool_name!r} wants to run with "
            f"{context.tool_input}. Allow? [y/N] "
        )
        return HookResult(should_continue=answer.strip().lower() == "y")

    hooks = HookRegistry()
    hooks.register(HookPoint.ON_PERMISSION_CHECK, cli_confirm)
    agent = Agent(llm=..., tools=[...], hooks=hooks)
"""

from kinetic_sdk.security.audit import AuditLogger, InMemoryAuditLogger, JSONLAuditLogger
from kinetic_sdk.security.policy import (
    AllowListPolicy,
    PermissionDecision,
    PermissionPolicy,
    PermissivePolicy,
)
from kinetic_sdk.security.redact import REDACTED, redact_secrets, redact_value

__all__ = [
    "AllowListPolicy",
    "AuditLogger",
    "InMemoryAuditLogger",
    "JSONLAuditLogger",
    "PermissionDecision",
    "PermissionPolicy",
    "PermissivePolicy",
    "REDACTED",
    "redact_secrets",
    "redact_value",
]
