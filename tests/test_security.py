"""Tests for the security package: policies, audit logging, redaction,
and their integration with the agent loop.
"""

from __future__ import annotations

from __future__ import annotations

import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.security import (
    REDACTED,
    AllowListPolicy,
    InMemoryAuditLogger,
    JSONLAuditLogger,
    PermissionDecision,
    PermissivePolicy,
    PolicyRule,
    RuleBasedPolicy,
    redact_secrets,
    redact_value,
)
from kinetic_sdk.tool.base import Tool, ToolResult
from tests._helpers import EchoTool, MockLLM, text_response, tool_response

NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)

FAKE_GHP = "ghp_" + "a1B2c3D4" * 5  # 40 chars after the prefix
FAKE_GHU = "ghu_" + "z9Y8x7W6" * 5
FAKE_PAT = "github_pat_" + "Q1w2E3r4T5" * 4

# --- AllowListPolicy -------------------------------------------------------

def test_allowlist_allows_listed_tool():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("echo", {"message": "hi"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False

def test_allowlist_denies_unlisted_tool_by_default():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("terminal", {"command": "ls"})
    assert decision.allowed is False
    assert decision.reason == "tool not in allow-list"

def test_allowlist_empty_by_default_denies_everything():
    policy = AllowListPolicy()
    assert policy.check("echo", {}).allowed is False

def test_allowlist_flags_dangerous_pattern_for_confirmation():
    policy = AllowListPolicy(
        always_allow=["terminal", "git"],
        require_confirmation_patterns={
            "terminal": ["rm -rf", "sudo"],
            "git": ["push"],
        },
    )
    decision = policy.check("terminal", {"command": "rm -rf /tmp/x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert "rm -rf" in decision.reason
    # Safe input on the same tool runs freely.
    assert policy.check("terminal", {"command": "ls -la"}).requires_confirmation is False
    # Patterns are per-tool.
    assert policy.check("git", {"args": ["status"]}).requires_confirmation is False
    assert policy.check("git", {"args": ["push", "origin", "main"]}).requires_confirmation is True

def test_allowlist_pattern_supports_regex():
    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": [r"rm\s+-[a-z]*r[a-z]*f"]},
    )
    assert policy.check("terminal", {"command": "rm   -rf /"}).requires_confirmation is True
    assert policy.check("terminal", {"command": "rm file.txt"}).requires_confirmation is False

# --- Permission Decision -----------------------------------------------------

def test_permission_decision_allowed():
    decision = PermissionDecision(allowed=True)
    assert decision.allowed is True
    assert decision.reason == ""

def test_permission_decision_denied():
    decision = PermissionDecision(allowed=False, reason="forbidden")
    assert decision.allowed is False
    assert decision.reason == "forbidden"

# --- Permission Policy -----------------------------------------------------

def test_allowlist_allows_listed_tool():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("echo", {"message": "hi"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False

def test_allowlist_denies_unlisted_tool_by_default():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("terminal", {"command": "ls"})
    assert decision.allowed is False
    assert decision.reason == "tool not in allow-list"

def test_allowlist_empty_by_default_denies_everything():
    policy = AllowListPolicy()
    assert policy.check("echo", {}).allowed is False

def test_allowlist_flags_dangerous_pattern_for_confirmation():
    policy = AllowListPolicy(
        always_allow=["terminal", "git"],
        require_confirmation_patterns={
            "terminal": ["rm -rf", "sudo"],
            "git": ["push"],
        },
    )
    decision = policy.check("terminal", {"command": "rm -rf /tmp/x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert "rm -rf" in decision.reason
    # Safe input on the same tool runs freely.
    assert policy.check("terminal", {"command": "ls -la"}).requires_confirmation is False
    # Patterns are per-tool.
    assert policy.check("git", {"args": ["status"]}).requires_confirmation is False
    assert policy.check("git", {"args": ["push", "origin", "main"]}).requires_confirmation is True

def test_allowlist_pattern_supports_regex():
    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": [r"rm\s+-[a-z]*r[a-z]*f"]},
    )
    assert policy.check("terminal", {"command": "rm   -rf /"}).requires_confirmation is True
    assert policy.check("terminal", {"command": "rm file.txt"}).requires_confirmation is False

# --- Permission Decision -----------------------------------------------------

def test_permission_decision_allowed():
    decision = PermissionDecision(allowed=True)
    assert decision.allowed is True
    assert decision.reason == ""

def test_permission_decision_denied():
    decision = PermissionDecision(allowed=False, reason="forbidden")
    assert decision.allowed is False
    assert decision.reason == "forbidden"

# --- Permission Policy -----------------------------------------------------

def test_allowlist_allows_listed_tool():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("echo", {"message": "hi"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False

def test_allowlist_denies_unlisted_tool_by_default():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("terminal", {"command": "ls"})
    assert decision.allowed is False
    assert decision.reason == "tool not in allow-list"

def test_allowlist_empty_by_default_denies_everything():
    policy = AllowListPolicy()
    assert policy.check("echo", {}).allowed is False

def test_allowlist_flags_dangerous_pattern_for_confirmation():
    policy = AllowListPolicy(
        always_allow=["terminal", "git"],
        require_confirmation_patterns={
            "terminal": ["rm -rf", "sudo"],
            "git": ["push"],
        },
    )
    decision = policy.check("terminal", {"command": "rm -rf /tmp/x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert "rm -rf" in decision.reason
    # Safe input on the same tool runs freely.
    assert policy.check("terminal", {"command": "ls -la"}).requires_confirmation is False
    # Patterns are per-tool.
    assert policy.check("git", {"args": ["status"]}).requires_confirmation is False
    assert policy.check("git", {"args": ["push", "origin", "main"]}).requires_confirmation is True

def test_allowlist_pattern_supports_regex():
    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": [r"rm\s+-[a-z]*r[a-z]*f"]},
    )
    assert policy.check("terminal", {"command": "rm   -rf /"}).requires_confirmation is True
    assert policy.check("terminal", {"command": "rm file.txt"}).requires_confirmation is False

# --- Permission Decision -----------------------------------------------------

def test_permission_decision_allowed():
    decision = PermissionDecision(allowed=True)
    assert decision.allowed is True
    assert decision.reason == ""

def test_permission_decision_denied():
    decision = PermissionDecision(allowed=False, reason="forbidden")
    assert decision.allowed is False
    assert decision.reason == "forbidden"

# --- Permission Policy -----------------------------------------------------

def test_allowlist_allows_listed_tool():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("echo", {"message": "hi"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False

def test_allowlist_denies_unlisted_tool_by_default():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("terminal", {"command": "ls"})
    assert decision.allowed is False
    assert decision.reason == "tool not in allow-list"

def test_allowlist_empty_by_default_denies_everything():
    policy = AllowListPolicy()
    assert policy.check("echo", {}).allowed is False

def test_allowlist_flags_dangerous_pattern_for_confirmation():
    policy = AllowListPolicy(
        always_allow=["terminal", "git"],
        require_confirmation_patterns={
            "terminal": ["rm -rf", "sudo"],
            "git": ["push"],
        },
    )
    decision = policy.check("terminal", {"command": "rm -rf /tmp/x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert "rm -rf" in decision.reason
    # Safe input on the same tool runs freely.
    assert policy.check("terminal", {"command": "ls -la"}).requires_confirmation is False
    # Patterns are per-tool.
    assert policy.check("git", {"args": ["status"]}).requires_confirmation is False
    assert policy.check("git", {"args": ["push", "origin", "main"]}).requires_confirmation is True

def test_allowlist_pattern_supports_regex():
    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": [r"rm\s+-[a-z]*r[a-z]*f"]},
    )
    assert policy.check("terminal", {"command": "rm   -rf /"}).requires_confirmation is True
    assert policy.check("terminal", {"command": "rm file.txt"}).requires_confirmation is False

# --- Audit Log Redactions ------------------------------------------------

def test_audit_loggers_redact_secrets_in_input_and_output():
    secret_input = {"command": f"curl -H 'token: {FAKE_GHP}' https://api.example.com"}
    logger = InMemoryAuditLogger()
    decision = PermissionDecision(allowed=True, reason="ok")
    logger.log_tool_call("terminal", secret_input, decision, NOW)
    logger.log_tool_result("terminal", ToolResult(output=f"authenticated as {FAKE_GHU}"), NOW)

    serialised = json.dumps(logger.entries, ensure_ascii=False)
    assert FAKE_GHP not in serialised
    assert FAKE_GHU not in serialised
    assert "[REDACTED]" in serialised

def test_audit_redacts_secrets_in_decision_reason_and_subject():
    # Test that policy reason and log_event subject are redacted
    policy = RuleBasedPolicy(
        [
            PolicyRule(
                tool_name="terminal",
                path_prefix="/workspace",
                decision=PermissionDecision(True, "tool in workspace"),
            ),
            PolicyRule(tool_name="terminal", decision=PermissionDecision(True, "tool in workspace")),
        ]
    )

    # Policy reason may contain secret-like text
    decision = PermissionDecision(
        allowed=True,
        reason="api_key=sk-abc123def456ghi789jkl012mno345pqr678",
        requires_confirmation=False
    )

    logger = InMemoryAuditLogger()
    logger.log_tool_call("terminal", {"message": "hi"}, decision, NOW)
    entry = logger.entries[-1]
    assert "api_key=sk-" not in entry["decision"]["reason"]

    logger.log_event("test_event", "api_key=sk-abc123def456ghi789jkl012mno345pqr678", NOW, key="value")
    entry = logger.entries[-1]
    assert "api_key=sk-" not in entry["tool_name"]  # subject is redacted in tool_name field

# --- Permission Decision -----------------------------------------------------

def test_permission_decision_allowed():
    decision = PermissionDecision(allowed=True)
    assert decision.allowed is True
    assert decision.reason == ""

def test_permission_decision_denied():
    decision = PermissionDecision(allowed=False, reason="forbidden")
    assert decision.allowed is False
    assert decision.reason == "forbidden"

# --- Permission Policy -----------------------------------------------------

def test_allowlist_allows_listed_tool():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("echo", {"message": "hi"})
    assert decision.allowed is True
    assert decision.requires_confirmation is False

def test_allowlist_denies_unlisted_tool_by_default():
    policy = AllowListPolicy(always_allow=["echo"])
    decision = policy.check("terminal", {"command": "ls"})
    assert decision.allowed is False
    assert decision.reason == "tool not in allow-list"

def test_allowlist_empty_by_default_denies_everything():
    policy = AllowListPolicy()
    assert policy.check("echo", {}).allowed is False

def test_allowlist_flags_dangerous_pattern_for_confirmation():
    policy = AllowListPolicy(
        always_allow=["terminal", "git"],
        require_confirmation_patterns={
            "terminal": ["rm -rf", "sudo"],
            "git": ["push"],
        },
    )
    decision = policy.check("terminal", {"command": "rm -rf /tmp/x"})
    assert decision.allowed is True
    assert decision.requires_confirmation is True
    assert "rm -rf" in decision.reason
    # Safe input on the same tool runs freely.
    assert policy.check("terminal", {"command": "ls -la"}).requires_confirmation is False
    # Patterns are per-tool.
    assert policy.check("git", {"args": ["status"]}).requires_confirmation is False
    assert policy.check("git", {"args": ["push", "origin", "main"]}).requires_confirmation is True

def test_allowlist_pattern_supports_regex():
    policy = AllowListPolicy(
        always_allow=["terminal"],
        require_confirmation_patterns={"terminal": [r"rm\s+-[a-z]*r[a-z]*f"]},
    )
    assert policy.check("terminal", {"command": "rm   -rf /"}).requires_confirmation is True
    assert policy.check("terminal", {"command": "rm file.txt"}).requires_confirmation is False

# --- Permission Decision -----------------------------------------------------

def test_permission_decision_allowed():
    decision = PermissionDecision(allowed=True)
    assert decision.allowed is True
    assert decision.reason == ""

def test_permission_decision_denied():
    decision = PermissionDecision(allowed=False, reason="forbidden")
    assert decision.allowed is False
    assert decision.reason == "forbidden"