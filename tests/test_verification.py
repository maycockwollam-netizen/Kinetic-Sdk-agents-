from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from kinetic_sdk.agent import Agent, AsyncAgent
from kinetic_sdk.observability import RunTrace
from kinetic_sdk.project import ProjectManifest, VerificationConfig
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response
from kinetic_sdk.testing.async_mocks import AsyncMockLLMClient
from kinetic_sdk.tool import ToolResult
from kinetic_sdk.verify import VerificationContract, VerificationStatus


def _event(kind, payload, second):
    return {"event_type": kind, "timestamp": datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc).isoformat(), "payload": payload}


def _manifest(kind="tests"):
    return ProjectManifest(test="python -m pytest -q", lint="ruff check kinetic_sdk", verification=VerificationConfig(kind=kind))


def test_verified_when_manifest_test_succeeds_after_last_edit():
    trace = RunTrace("run", [
        _event("agent.tool_call_started", {"name": "apply_patch", "arguments": {"patch": "*** Update File: a.py"}}, 1),
        _event("agent.tool_call_finished", {"name": "apply_patch", "output": "Done"}, 2),
        _event("agent.tool_call_started", {"name": "terminal", "arguments": {"command": "python -m pytest -q"}}, 3),
        _event("agent.tool_call_finished", {"name": "terminal", "output": "2 passed", "metadata": {"exit_code": 0}}, 4),
    ])
    contract = VerificationContract.from_trace(trace, _manifest())
    assert contract.status is VerificationStatus.VERIFIED
    assert contract.changed_files == ["a.py"]


def test_edit_after_success_is_unverified():
    trace = RunTrace("run", [
        _event("agent.tool_call_started", {"name": "terminal", "arguments": {"command": "python -m pytest -q"}}, 1),
        _event("agent.tool_call_finished", {"name": "terminal", "output": "2 passed", "metadata": {"exit_code": 0}}, 2),
        _event("agent.tool_call_started", {"name": "file", "arguments": {"action": "write", "path": "a.py"}}, 3),
    ])
    contract = VerificationContract.from_trace(trace, _manifest())
    assert contract.status is VerificationStatus.UNVERIFIED
    assert contract.edits_after_last_verification


def test_failed_verification_command_is_failed():
    trace = RunTrace("run", [
        _event("agent.tool_call_started", {"name": "terminal", "arguments": {"command": "python -m pytest -q"}}, 1),
        _event("agent.tool_call_finished", {"name": "terminal", "output": "1 failed", "metadata": {"exit_code": 1}}, 2),
    ])
    assert VerificationContract.from_trace(trace, _manifest()).status is VerificationStatus.FAILED


def test_claims_not_present_in_trace_are_reported():
    trace = RunTrace("run", [])
    contract = VerificationContract.from_trace(trace, _manifest(), '{"changed_files":["fake.py"],"commands_run":[{"command":"pytest","exit_code":0}]}')
    assert contract.claims_mismatch
    assert contract.status is VerificationStatus.UNVERIFIED


def test_lint_and_diff_strategies():
    lint = RunTrace("run", [
        _event("agent.tool_call_started", {"name": "terminal", "arguments": {"command": "ruff check kinetic_sdk"}}, 1),
        _event("agent.tool_call_finished", {"name": "terminal", "metadata": {"exit_code": 0}}, 2),
    ])
    assert VerificationContract.from_trace(lint, _manifest("lint")).status is VerificationStatus.VERIFIED
    diff_manifest = ProjectManifest(max_diff_lines=3, verification=VerificationConfig(kind="diff"))
    diff = RunTrace("run", [_event("agent.tool_call_finished", {"name": "git", "output": "a\nb\nc"}, 1)])
    assert VerificationContract.from_trace(diff, diff_manifest).status is VerificationStatus.VERIFIED


def test_agent_opt_in_prefixes_unverified_and_publishes_contract():
    manifest = _manifest()
    llm = MockLLMClient([tool_response("one", "terminal", {"command": "python -m pytest -q"}), text_response("done")])
    tool = MockTool("terminal", result=ToolResult(output="1 passed", metadata={"exit_code": 0}))
    agent = Agent(llm, [tool], permission_policy=PermissivePolicy(), project_manifest=manifest, require_verification=True)
    assert agent.run("go") == "done"
    assert agent.last_verification is not None
    assert agent.last_verification.status is VerificationStatus.VERIFIED


def test_async_agent_verification_parity():
    manifest = _manifest()
    llm = AsyncMockLLMClient([tool_response("one", "terminal", {"command": "python -m pytest -q"}), text_response("done")])
    tool = MockTool("terminal", result=ToolResult(output="1 passed", metadata={"exit_code": 0}))
    agent = AsyncAgent(llm, [tool], permission_policy=PermissivePolicy(), project_manifest=manifest, require_verification=True)
    assert asyncio.run(agent.run("go")) == "done"
    assert agent.last_verification is not None
    assert agent.last_verification.status is VerificationStatus.VERIFIED
