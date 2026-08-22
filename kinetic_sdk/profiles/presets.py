"""Agent configuration presets (Stage 4, part C).

Each factory returns a plain ``dict`` of keyword arguments ready to splat
into :class:`~kinetic_sdk.agent.agent.Agent`::

    from kinetic_sdk.profiles import dev_profile

    agent = Agent(llm=client, tools=[...], **dev_profile())

The factories deliberately do NOT construct the ``Agent`` themselves: when
(and whether) an agent is instantiated is the SDK user's decision, and
importing this module must stay free of side effects.

⚠️ These presets are *starting points*, not the one true configuration. Read
each docstring, then adjust the returned dict (or write your own factory) to
match your real environment.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from kinetic_sdk.context.manager import SummarizingContextManager
from kinetic_sdk.git.tool import GitTool
from kinetic_sdk.llm.client import LLMClient
from kinetic_sdk.observability.logger import ConsoleObservabilityLogger
from kinetic_sdk.security.audit import InMemoryAuditLogger, JSONLAuditLogger
from kinetic_sdk.security.policy import AllowListPolicy, PermissivePolicy


def dev_profile() -> dict[str, Any]:
    """Preset for local development / quick experimentation.

    * :class:`PermissivePolicy` — every tool runs without checks. Convenient
      on a throwaway checkout, dangerous anywhere real. ⚠️ KHÔNG dùng trong
      production.
    * :class:`InMemoryAuditLogger` — audit entries kept in memory for
      inspection while debugging.
    * :class:`ConsoleObservabilityLogger` — every agent event printed live,
      so you can watch the loop without extra setup.

    Context management and classification stay at the ``Agent`` defaults
    (truncation-based compaction, MAX routing via the default classifier).
    """
    return {
        "permission_policy": PermissivePolicy(),
        "audit_logger": InMemoryAuditLogger(),
        "observability_logger": ConsoleObservabilityLogger(),
    }


def production_profile(
    *,
    audit_log_path: str | os.PathLike[str],
    allowed_tools: Iterable[str] | None = None,
    summarizer_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Preset with strict defaults for unattended/production runs.

    * :class:`AllowListPolicy` — deny-by-default: only the tools named in
      ``allowed_tools`` may run at all. When ``"git"`` is allowed, the policy
      is pre-loaded with :attr:`GitTool.REQUIRE_CONFIRMATION_PATTERNS` so a
      force push always needs human confirmation (wire an
      ``ON_PERMISSION_CHECK`` hook to actually approve it — with no hook the
      call is denied, the safe default).
    * :class:`JSONLAuditLogger` — every tool call, denial and result appended
      to ``audit_log_path`` as JSON lines for later forensics.
    * :class:`SummarizingContextManager` — only when ``summarizer_client`` is
      provided (a cheap model client is recommended, mirroring the classifier
      pattern); otherwise the key is omitted and the ``Agent`` default
      (truncation) applies.

    Args:
        audit_log_path: File the audit log is appended to (created if
            missing, one JSON object per line).
        allowed_tools: Tool names the policy allows. Empty/omitted means the
            agent can call no tools until you extend the list — that is
            intentional.
        summarizer_client: Optional LLM client used to summarise compacted
            context spans.
    """
    allowed = list(allowed_tools or [])
    confirmation_patterns: dict[str, list[str]] = {}
    if "git" in allowed:
        confirmation_patterns["git"] = list(GitTool.REQUIRE_CONFIRMATION_PATTERNS)
    profile: dict[str, Any] = {
        "permission_policy": AllowListPolicy(
            always_allow=allowed,
            require_confirmation_patterns=confirmation_patterns,
        ),
        "audit_logger": JSONLAuditLogger(audit_log_path),
    }
    if summarizer_client is not None:
        profile["context_manager"] = SummarizingContextManager(
            summarizer_client=summarizer_client
        )
    return profile
