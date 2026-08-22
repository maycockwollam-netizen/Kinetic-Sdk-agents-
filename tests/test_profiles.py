"""Tests for the profiles package: presets must produce dicts that splat
cleanly into Agent(...) and carry the promised components.
"""

from __future__ import annotations

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.context.manager import SummarizingContextManager
from kinetic_sdk.observability.logger import ConsoleObservabilityLogger
from kinetic_sdk.profiles import dev_profile, production_profile
from kinetic_sdk.security import (
    AllowListPolicy,
    InMemoryAuditLogger,
    JSONLAuditLogger,
    PermissivePolicy,
)
from tests._helpers import EchoTool, MockLLM, text_response, tool_response


# --- dev_profile ----------------------------------------------------------------


def test_dev_profile_components():
    profile = dev_profile()
    assert isinstance(profile["permission_policy"], PermissivePolicy)
    assert isinstance(profile["audit_logger"], InMemoryAuditLogger)
    assert isinstance(profile["observability_logger"], ConsoleObservabilityLogger)


def test_dev_profile_builds_working_agent():
    agent = Agent(llm=MockLLM([text_response("ok")]), tools=[EchoTool()], **dev_profile())
    assert agent.run("hello") == "ok"


def test_dev_profile_allows_tools_out_of_the_box(tmp_path):
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": "hi"}),
            text_response("done"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], **dev_profile())
    assert agent.run("say hi") == "done"


# --- production_profile ----------------------------------------------------------


def test_production_profile_components(tmp_path):
    profile = production_profile(audit_log_path=tmp_path / "audit.jsonl")
    assert isinstance(profile["permission_policy"], AllowListPolicy)
    assert isinstance(profile["audit_logger"], JSONLAuditLogger)
    # No summarizer client -> key omitted, Agent default (truncation) applies.
    assert "context_manager" not in profile


def test_production_profile_denies_unlisted_tools_by_default(tmp_path):
    profile = production_profile(audit_log_path=tmp_path / "audit.jsonl")
    decision = profile["permission_policy"].check("echo", {"message": "hi"})
    assert decision.allowed is False


def test_production_profile_marks_git_force_push_for_confirmation(tmp_path):
    profile = production_profile(
        audit_log_path=tmp_path / "audit.jsonl", allowed_tools=["git"]
    )
    policy = profile["permission_policy"]
    force = policy.check("git", {"action": "push", "branch": "main", "force": True})
    assert force.allowed is True
    assert force.requires_confirmation is True
    normal = policy.check("git", {"action": "status"})
    assert normal.allowed is True
    assert normal.requires_confirmation is False


def test_production_profile_with_summarizer_client(tmp_path):
    profile = production_profile(
        audit_log_path=tmp_path / "audit.jsonl",
        allowed_tools=["echo"],
        summarizer_client=MockLLM([]),
    )
    assert isinstance(profile["context_manager"], SummarizingContextManager)


def test_production_profile_builds_working_agent_and_audits(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    profile = production_profile(audit_log_path=audit_path, allowed_tools=["echo"])
    llm = MockLLM(
        [
            tool_response("c1", "echo", {"message": "hi"}),
            text_response("done"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], **profile)
    assert agent.run("say hi") == "done"
    # The JSONL audit logger actually wrote the run to disk.
    assert audit_path.read_text().strip() != ""


def test_profiles_do_not_construct_agents_eagerly():
    # Factories return plain dicts; instantiating Agent stays the caller's
    # decision (no hidden side effects at import/call time).
    assert isinstance(dev_profile(), dict)
    assert all(isinstance(k, str) for k in dev_profile())
