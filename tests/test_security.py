"""Tests for the security package: policies, audit logging, redaction,
and their integration with the agent loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.security import (
    AllowListPolicy,
    InMemoryAuditLogger,
    JSONLAuditLogger,
    PermissionDecision,
    PermissivePolicy,
    redact_secrets,
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
    assert AllowListPolicy().check("echo", {}).allowed is False


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


# --- PermissivePolicy ------------------------------------------------------


def test_permissive_policy_allows_anything():
    policy = PermissivePolicy()
    assert policy.check("anything", {}).allowed is True
    assert policy.check("terminal", {"command": "rm -rf /"}).allowed is True


def test_permissive_policy_never_requires_confirmation():
    assert PermissivePolicy().check("git", {"args": ["push"]}).requires_confirmation is False


# --- redact_secrets --------------------------------------------------------


def test_redact_github_tokens():
    for token in (FAKE_GHP, FAKE_GHU, FAKE_PAT):
        redacted = redact_secrets(f"fetching with {token} done")
        assert token not in redacted
        assert "[REDACTED]" in redacted


def test_redact_keyword_followed_by_long_token():
    text = f'api_key = "{"Ab3" * 10}x9" and token: {"Zz9" * 8}'
    redacted = redact_secrets(text)
    assert "Ab3Ab3" not in redacted
    assert "Zz9Zz9" not in redacted
    assert "api_key" in redacted  # the keyword itself is not secret


def test_redact_leaves_normal_text_untouched():
    text = "echo hello world; short_key=abc; id=12345"
    assert redact_secrets(text) == text


def test_redact_empty_and_short_inputs():
    assert redact_secrets("") == ""
    assert redact_secrets("hi") == "hi"


# --- Audit loggers ---------------------------------------------------------


def test_in_memory_audit_logger_records_all_entry_types_in_order():
    logger = InMemoryAuditLogger()
    decision = PermissionDecision(allowed=True, reason="tool in allow-list")
    logger.log_tool_call("echo", {"message": "hi"}, decision, NOW)
    logger.log_tool_result("echo", ToolResult(output="hi"), NOW)
    logger.log_permission_denied("terminal", {"command": "ls"}, "tool not in allow-list", NOW)

    assert [e["event"] for e in logger.entries] == [
        "tool_call",
        "tool_result",
        "permission_denied",
    ]
    call = logger.entries[0]
    assert call["tool_name"] == "echo"
    assert call["input"] == {"message": "hi"}
    assert call["decision"] == {
        "allowed": True,
        "reason": "tool in allow-list",
        "requires_confirmation": False,
    }
    result = logger.entries[1]
    assert result["result"]["is_error"] is False
    assert result["result"]["output"] == "hi"
    assert logger.entries[2]["reason"] == "tool not in allow-list"
    # Every entry has a unique id and an ISO timestamp.
    ids = {e["id"] for e in logger.entries}
    assert len(ids) == 3
    assert all(e["timestamp"] == NOW.isoformat() for e in logger.entries)


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


def test_jsonl_audit_logger_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    decision = PermissionDecision(allowed=False, reason="tool not in allow-list")
    with JSONLAuditLogger(path) as logger:
        logger.log_tool_call("rm_tool", {"target": f"secret {FAKE_PAT}"}, decision, NOW)
        logger.log_permission_denied("rm_tool", {"target": f"secret {FAKE_PAT}"}, "denied", NOW)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]  # each line parses on its own
    assert [e["event"] for e in entries] == ["tool_call", "permission_denied"]
    raw = path.read_text(encoding="utf-8")
    assert FAKE_PAT not in raw


# --- Agent integration -----------------------------------------------------


class SpyTool(Tool):
    """Counts executions so tests can prove a denied tool never ran."""

    name = "spy"
    description = "Records how many times it was executed."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executions = 0

    def execute(self, **params: Any) -> ToolResult:
        self.executions += 1
        return ToolResult(output="executed")


def test_agent_default_policy_is_deny_by_default():
    agent = Agent(llm=MockLLM([text_response("ok")]))
    assert isinstance(agent.permission_policy, AllowListPolicy)
    assert agent.permission_policy.always_allow == frozenset()
    assert isinstance(agent.audit_logger, InMemoryAuditLogger)


def test_agent_denied_tool_never_executes_and_model_sees_error():
    spy = SpyTool()
    llm = MockLLM([tool_response("c1", "spy", {}), text_response("gave up")])
    bus = EventBus()
    denied_events: list[Event] = []
    bus.subscribe("security.permission_denied", denied_events.append)
    agent = Agent(llm=llm, tools=[spy], event_bus=bus)  # default: deny-all policy

    assert agent.run("do it") == "gave up"
    assert spy.executions == 0

    # The denial event was emitted with tool identity and reason.
    assert len(denied_events) == 1
    assert denied_events[0].payload["name"] == "spy"
    assert "tool not in allow-list" in denied_events[0].payload["reason"]

    # The model received an error tool_result explaining the denial.
    tool_results = [
        b for m in agent.state.messages if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert tool_results and tool_results[0]["is_error"] is True
    assert "Permission denied" in json.dumps(tool_results[0]["content"])

    # Both the call and the denial were audited, in order.
    events = [e["event"] for e in agent.audit_logger.entries]
    assert events == ["tool_call", "permission_denied"]


def test_agent_requires_confirmation_is_denied_in_automated_mode():
    spy = SpyTool()
    policy = AllowListPolicy(
        always_allow=["spy"],
        require_confirmation_patterns={"spy": ["destroy"]},
    )
    llm = MockLLM(
        [tool_response("c1", "spy", {"target": "destroy"}), text_response("blocked")]
    )
    bus = EventBus()
    denied_events: list[Event] = []
    bus.subscribe("security.permission_denied", denied_events.append)
    agent = Agent(llm=llm, tools=[spy], permission_policy=policy, event_bus=bus)

    assert agent.run("go") == "blocked"
    assert spy.executions == 0
    reason = denied_events[0].payload["reason"]
    assert "requires manual confirmation" in reason
    assert "not yet supported" in reason


def test_agent_allowed_tool_executes_and_is_audited():
    llm = MockLLM([tool_response("c1", "echo", {"message": "hi"}), text_response("done")])
    policy = AllowListPolicy(always_allow=["echo"])
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=policy)

    assert agent.run("say hi") == "done"
    events = [e["event"] for e in agent.audit_logger.entries]
    assert events == ["tool_call", "tool_result"]
    entry = agent.audit_logger.entries[1]
    assert entry["tool_name"] == "echo"
    assert entry["result"]["output"] == "hi"
    assert entry["result"]["is_error"] is False


def test_agent_audit_log_contains_no_raw_secrets():
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": f"my token is {FAKE_GHP}"}),
            text_response("done"),
        ]
    )
    policy = AllowListPolicy(always_allow=["echo"])
    agent = Agent(llm=llm, tools=[EchoTool()], permission_policy=policy)
    agent.run("leak")

    serialised = json.dumps(agent.audit_logger.entries, ensure_ascii=False)
    assert FAKE_GHP not in serialised
    assert "[REDACTED]" in serialised


def test_agent_denial_event_payload_is_redacted():
    llm = MockLLM([tool_response("c1", "spy", {"auth": FAKE_GHU}), text_response("x")])
    bus = EventBus()
    denied_events: list[Event] = []
    bus.subscribe("security.permission_denied", denied_events.append)
    agent = Agent(llm=llm, tools=[SpyTool()], event_bus=bus)
    agent.run("go")

    payload = json.dumps(denied_events[0].payload, ensure_ascii=False)
    assert FAKE_GHU not in payload
    assert "[REDACTED]" in payload
