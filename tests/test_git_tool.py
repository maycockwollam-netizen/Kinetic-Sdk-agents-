"""Tests for the git package: GitTool sub-actions, credential handling and
force-push marking. No real git process is ever spawned — a fake runner
scripts each command's output.
"""

from __future__ import annotations

import json
import subprocess

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.git import GitTool
from kinetic_sdk.secret import DictSecretProvider, SecretProvider, SecretRegistry
from kinetic_sdk.security import AllowListPolicy, PermissivePolicy
from tests._helpers import MockLLM, text_response, tool_response

FAKE_TOKEN = "ghp_" + "t0k3nV4lu3" * 4  # matches the ghp_ redaction pattern


class FakeGitRunner:
    """Scripted stand-in for subprocess: records argv, replays responses.

    Each response is a ``(returncode, stdout, stderr)`` tuple or a callable
    ``(argv) -> CompletedProcess``. Without a scripted response the command
    "succeeds" with output ``"ok"``.
    """

    def __init__(self, responses: list | None = None) -> None:
        self.calls: list[list[str]] = []
        self._responses = list(responses or [])

    def __call__(self, argv: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if self._responses:
            response = self._responses.pop(0)
            if callable(response):
                return response(argv)
            returncode, stdout, stderr = response
            return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
        return subprocess.CompletedProcess(argv, 0, "ok", "")


class SpyProvider(SecretProvider):
    """Records which keys were requested (verifies registry-based lookup)."""

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = dict(secrets)
        self.requested: list[str] = []

    def get(self, key: str) -> str | None:
        self.requested.append(key)
        return self._secrets.get(key)


def make_tool(responses: list | None = None, **kwargs) -> tuple[GitTool, FakeGitRunner]:
    runner = FakeGitRunner(responses)
    kwargs.setdefault("workdir", "/repo")
    return GitTool(runner=runner, **kwargs), runner


# --- Sub-action argv construction -------------------------------------------


def test_status_runs_short_branch_summary():
    tool, runner = make_tool(responses=[(0, "## main\n M file.py\n", "")])
    result = tool.execute(action="status")
    assert not result.is_error
    assert runner.calls[0] == ["git", "status", "--short", "--branch"]
    assert "M file.py" in result.output


def test_diff_unstaged_and_staged():
    tool, runner = make_tool()
    tool.execute(action="diff")
    tool.execute(action="diff", staged=True)
    assert runner.calls[0] == ["git", "diff"]
    assert runner.calls[1] == ["git", "diff", "--staged"]


def test_add_requires_paths_and_terminates_options():
    tool, runner = make_tool()
    result = tool.execute(action="add", paths=[])
    assert result.is_error
    assert runner.calls == []

    tool.execute(action="add", paths=["a.py", "-f"])
    # "--" keeps even "-f" treated as a path, never as a flag.
    assert runner.calls[0] == ["git", "add", "--", "a.py", "-f"]


def test_commit_requires_message():
    tool, runner = make_tool()
    for bad in ("", "   ", None):
        result = tool.execute(action="commit", message=bad)
        assert result.is_error
    assert runner.calls == []

    tool.execute(action="commit", message="feat: add git tool")
    assert runner.calls[0] == ["git", "commit", "-m", "feat: add git tool"]


def test_branch_lists_or_creates():
    tool, runner = make_tool()
    tool.execute(action="branch")
    tool.execute(action="branch", branch="feature/x")
    assert runner.calls[0] == ["git", "branch", "--list"]
    assert runner.calls[1] == ["git", "branch", "feature/x"]


def test_checkout_switches_or_creates():
    tool, runner = make_tool()
    tool.execute(action="checkout", branch="main")
    tool.execute(action="checkout", branch="feature/y", create=True)
    assert runner.calls[0] == ["git", "checkout", "main"]
    assert runner.calls[1] == ["git", "checkout", "-b", "feature/y"]


def test_option_injection_in_refs_is_rejected():
    tool, runner = make_tool()
    for kwargs in (
        {"action": "branch", "branch": "--delete"},
        {"action": "checkout", "branch": "--force"},
        {"action": "push", "branch": "--force"},
        {"action": "push", "branch": "main", "remote": "--upload-pack=evil"},
    ):
        result = tool.execute(**kwargs)
        assert result.is_error, kwargs
        assert "invalid" in result.error
    assert runner.calls == []


def test_log_limits_history_length():
    tool, runner = make_tool()
    tool.execute(action="log")
    tool.execute(action="log", max_count=5)
    tool.execute(action="log", max_count=10_000)
    assert runner.calls[0] == ["git", "log", "--oneline", "--max-count=20"]
    assert runner.calls[1] == ["git", "log", "--oneline", "--max-count=5"]
    assert runner.calls[2] == ["git", "log", "--oneline", "--max-count=100"]


def test_unknown_action_is_rejected():
    tool, runner = make_tool()
    result = tool.execute(action="rebase")
    assert result.is_error
    assert "rebase" in result.error
    assert runner.calls == []


def test_unexpected_parameter_is_reported():
    tool, runner = make_tool()
    result = tool.execute(action="status", bogus=1)
    assert result.is_error
    assert "Invalid parameters" in result.error
    assert runner.calls == []


def test_git_failure_surfaces_stderr():
    tool, _ = make_tool(responses=[(128, "", "fatal: not a git repository")])
    result = tool.execute(action="status")
    assert result.is_error
    assert "fatal: not a git repository" in result.error


# --- Credential handling ------------------------------------------------------


def test_push_uses_token_from_secret_registry():
    provider = SpyProvider({"GIT_TOKEN": FAKE_TOKEN})
    secrets = SecretRegistry([provider])
    tool, runner = make_tool(secrets=secrets)
    result = tool.execute(action="push", branch="main")

    assert not result.is_error
    assert provider.requested == ["GIT_TOKEN"]
    argv = runner.calls[0]
    assert argv[:2] == ["git", "-c"]
    assert f"http.extraHeader=AUTHORIZATION: bearer {FAKE_TOKEN}" in argv
    assert argv[-3:] == ["push", "origin", "main"]


def test_push_without_token_runs_without_credential_header():
    secrets = SecretRegistry([DictSecretProvider({})])
    tool, runner = make_tool(secrets=secrets)
    result = tool.execute(action="push", branch="main")
    assert not result.is_error
    assert runner.calls[0] == ["git", "push", "origin", "main"]


def test_token_is_scrubbed_from_result_output_and_metadata():
    # git echoes the credential back on stderr (e.g. in an error URL/header).
    responses = [(1, "", f"fatal: auth failed for bearer {FAKE_TOKEN}")]
    secrets = SecretRegistry([DictSecretProvider({"GIT_TOKEN": FAKE_TOKEN})])
    tool, _ = make_tool(responses=responses, secrets=secrets)
    result = tool.execute(action="push", branch="main")

    assert result.is_error
    assert FAKE_TOKEN not in result.error
    assert "[REDACTED]" in result.error
    assert FAKE_TOKEN not in json.dumps(result.metadata)


def test_pull_also_resolves_credential():
    provider = SpyProvider({"GIT_TOKEN": FAKE_TOKEN})
    tool, runner = make_tool(secrets=SecretRegistry([provider]))
    tool.execute(action="pull", remote="upstream", branch="dev")
    assert provider.requested == ["GIT_TOKEN"]
    argv = runner.calls[0]
    assert argv[-3:] == ["pull", "upstream", "dev"]


def test_custom_token_key_is_used():
    provider = SpyProvider({"MY_GIT_PAT": FAKE_TOKEN})
    tool, runner = make_tool(
        secrets=SecretRegistry([provider]), remote_token_key="MY_GIT_PAT"
    )
    tool.execute(action="push", branch="main")
    assert provider.requested == ["MY_GIT_PAT"]
    assert FAKE_TOKEN in runner.calls[0][2]


# --- Force-push marking & permission policy -----------------------------------


def test_force_push_builds_force_flag_and_marks_metadata():
    tool, runner = make_tool()
    result = tool.execute(action="push", branch="main", force=True)
    assert runner.calls[0][-1] == "--force"
    assert result.metadata["force"] is True
    assert result.metadata["action"] == "push"


def test_force_push_input_matches_confirmation_patterns():
    policy = AllowListPolicy(
        always_allow=["git"],
        require_confirmation_patterns={"git": GitTool.REQUIRE_CONFIRMATION_PATTERNS},
    )
    force_decision = policy.check("git", {"action": "push", "branch": "main", "force": True})
    assert force_decision.allowed is True
    assert force_decision.requires_confirmation is True

    normal_decision = policy.check("git", {"action": "push", "branch": "main"})
    assert normal_decision.allowed is True
    assert normal_decision.requires_confirmation is False


# --- Agent-loop integration ----------------------------------------------------


def test_git_tool_flows_through_agent_permission_check():
    runner = FakeGitRunner(responses=[(0, "## main\n", "")])
    tool = GitTool(workdir="/repo", runner=runner)
    llm = MockLLM(
        [
            tool_response("c1", "git", {"action": "status"}),
            text_response("done"),
        ]
    )
    agent = Agent(llm=llm, tools=[tool], permission_policy=PermissivePolicy())
    assert agent.run("check the repo") == "done"
    assert runner.calls[0] == ["git", "status", "--short", "--branch"]


def test_git_tool_denied_by_default_policy():
    runner = FakeGitRunner()
    tool = GitTool(workdir="/repo", runner=runner)
    llm = MockLLM(
        [
            tool_response("c1", "git", {"action": "status"}),
            text_response("denied, stopping"),
        ]
    )
    # Default AllowListPolicy (empty) must deny git like any other tool —
    # GitTool does not bypass the permission system.
    agent = Agent(llm=llm, tools=[tool])
    assert agent.run("check the repo") == "denied, stopping"
    assert runner.calls == []
