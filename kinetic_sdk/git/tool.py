"""Git operations as a first-class agent tool (Stage 4, part A).

:class:`GitTool` wraps the handful of git operations a coding agent needs
day-to-day (``status``, ``diff``, ``add``, ``commit``, ``branch``,
``checkout``, ``push``, ``pull``, ``log``) behind the standard
:class:`~kinetic_sdk.tool.base.Tool` interface, instead of letting the agent
shell out to git through a generic terminal tool. Benefits over raw terminal
access:

* **Controlled surface** — only the declared sub-actions exist; arguments are
  validated and always passed as an argv list (never ``shell=True``), and
  values that look like options (``--force`` as a branch name, ...) are
  rejected before they reach git.
* **Credential lifecycle** — the token used for remote operations
  (``push``/``pull``) is resolved through
  :class:`~kinetic_sdk.secret.registry.SecretRegistry` and only revealed
  (``SecretValue.reveal()``) at the moment the command line is built. The plaintext
  token is scrubbed from any output before it is returned in a
  :class:`~kinetic_sdk.tool.base.ToolResult`.
* **Auditability** — the tool is an ordinary ``Tool``: every call still flows
  through the agent's ``permission_policy`` and audit logger exactly like any
  other tool. Nothing here bypasses :mod:`kinetic_sdk.security`.

Force-push safety
-----------------
A ``push`` with ``force=True`` is dangerous (it rewrites remote history).
The danger is visible to the permission layer *through the tool input*:
the JSON-serialised input contains ``"force": true``, and
:attr:`GitTool.REQUIRE_CONFIRMATION_PATTERNS` ships a ready-made pattern list
for ``AllowListPolicy`` so SDK users can force a confirmation step::

    from kinetic_sdk.git import GitTool
    from kinetic_sdk.security import AllowListPolicy

    policy = AllowListPolicy(
        always_allow=["git"],
        require_confirmation_patterns={"git": GitTool.REQUIRE_CONFIRMATION_PATTERNS},
    )

The executed :class:`ToolResult` also carries ``metadata["force"] = True`` so
audit entries record that a force push happened.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, ClassVar

from kinetic_sdk.secret.registry import SecretRegistry
from kinetic_sdk.security.redact import REDACTED, redact_secrets
from kinetic_sdk.tool.base import Tool, ToolResult

#: Signature of the command runner used by :class:`GitTool`. Receives the full
#: argv (starting with ``"git"``), the working directory and a timeout in
#: seconds, and returns a completed process. Injectable so tests never spawn
#: a real git process.
GitRunner = Callable[[list[str], str, float], "subprocess.CompletedProcess[str]"]


class GitTool(Tool):
    """Run a curated set of git operations inside one working directory.

    Args:
        workdir: Directory every git command runs in. Defaults to the current
            working directory at construction time. Pair it with
            :class:`kinetic_sdk.workspace.manager.Workspace` (``workdir =
            workspace.root_path``) when the agent is scoped to a workspace.
        secrets: Registry used to resolve the remote credential for
            ``push``/``pull``. Defaults to the environment-backed registry;
            the module never reads ``os.environ`` directly.
        remote_token_key: Key passed to ``secrets.resolve(...)`` when a remote
            operation needs a token. Missing token is not an error — git then
            falls back to its own credential configuration.
        timeout: Per-command timeout in seconds.
        runner: Command runner, defaults to :func:`subprocess.run`. Tests
            inject a fake here so no real git process is spawned.

    The tool intentionally does NOT cover every git command (no rebase,
    cherry-pick, submodule, ...). Anything outside :attr:`ACTIONS` is rejected
    before a process is spawned.
    """

    name: ClassVar[str] = "git"
    description: ClassVar[str] = (
        "Run git operations in the project repository. Exactly one 'action' "
        "per call: 'status' (working-tree summary), 'diff' (unstaged changes, "
        "or staged=true for staged), 'add' (stage 'paths'), 'commit' (requires "
        "'message'), 'branch' (list, or create when 'branch' given), "
        "'checkout' (switch to 'branch', or create+switch with create=true), "
        "'push' (requires 'branch'; 'remote' defaults to origin; force=true "
        "rewrites remote history and may require human confirmation), 'pull' "
        "('remote'/'branch' optional), 'log' (recent commits, 'max_count' "
        "capped)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status",
                    "diff",
                    "add",
                    "commit",
                    "branch",
                    "checkout",
                    "push",
                    "pull",
                    "log",
                ],
                "description": "The git sub-operation to run.",
            },
            "message": {
                "type": "string",
                "description": "Commit message. Required for action='commit'.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to stage. Required for action='add'.",
            },
            "branch": {
                "type": "string",
                "description": (
                    "Branch name. Required for 'checkout' and 'push'; creates "
                    "a branch for action='branch'; optional for 'pull'."
                ),
            },
            "create": {
                "type": "boolean",
                "default": False,
                "description": "With action='checkout', create the branch first (-b).",
            },
            "staged": {
                "type": "boolean",
                "default": False,
                "description": "With action='diff', show staged changes instead of unstaged.",
            },
            "remote": {
                "type": "string",
                "default": "origin",
                "description": "Remote name for 'push'/'pull'.",
            },
            "force": {
                "type": "boolean",
                "default": False,
                "description": (
                    "With action='push', force-update the remote ref. "
                    "DANGEROUS: rewrites remote history; permission policies "
                    "should require confirmation (see "
                    "GitTool.REQUIRE_CONFIRMATION_PATTERNS)."
                ),
            },
            "max_count": {
                "type": "integer",
                "default": 20,
                "description": "Maximum commits returned by action='log' (capped at 100).",
            },
        },
        "required": ["action"],
    }

    #: Sub-actions understood by :meth:`execute`.
    ACTIONS: ClassVar[tuple[str, ...]] = (
        "status",
        "diff",
        "add",
        "commit",
        "branch",
        "checkout",
        "push",
        "pull",
        "log",
    )

    #: Ready-made ``require_confirmation_patterns`` entry for
    #: :class:`~kinetic_sdk.security.policy.AllowListPolicy`. The serialised
    #: tool input of a force push contains ``"force": true``, so a policy
    #: configured with these patterns flags it ``requires_confirmation``
    #: instead of letting it run unattended.
    REQUIRE_CONFIRMATION_PATTERNS: ClassVar[list[str]] = ['"force": true']

    #: Default/capped number of commits returned by the ``log`` action, so a
    #: long history can never flood the conversation.
    DEFAULT_LOG_COUNT: ClassVar[int] = 20
    MAX_LOG_COUNT: ClassVar[int] = 100

    DEFAULT_REMOTE: ClassVar[str] = "origin"

    #: Default secret key holding the remote credential.
    DEFAULT_TOKEN_KEY: ClassVar[str] = "GIT_TOKEN"

    def __init__(
        self,
        workdir: str | None = None,
        secrets: SecretRegistry | None = None,
        remote_token_key: str = DEFAULT_TOKEN_KEY,
        timeout: float = 60.0,
        runner: GitRunner | None = None,
    ) -> None:
        self._workdir = workdir if workdir is not None else os.getcwd()
        self._secrets = secrets if secrets is not None else SecretRegistry()
        self._remote_token_key = remote_token_key
        self._timeout = float(timeout)
        self._runner: GitRunner = runner if runner is not None else self._subprocess_runner

    # -- Tool interface -----------------------------------------------------

    def execute(self, action: str, **params: Any) -> ToolResult:
        """Dispatch one git sub-action after validating its parameters."""
        handler = getattr(self, f"_action_{action}", None)
        if action not in self.ACTIONS or handler is None:
            return ToolResult(
                error=f"Unknown git action {action!r}. Supported: {', '.join(self.ACTIONS)}"
            )
        try:
            return handler(**params)
        except TypeError as exc:
            return ToolResult(error=f"Invalid parameters for git action {action!r}: {exc}")

    # -- Sub-actions ----------------------------------------------------------

    def _action_status(self) -> ToolResult:
        return self._run_git(["status", "--short", "--branch"])

    def _action_diff(self, staged: bool = False) -> ToolResult:
        argv = ["diff"] + (["--staged"] if staged else [])
        return self._run_git(argv)

    def _action_add(self, paths: list[str]) -> ToolResult:
        if not paths:
            return ToolResult(error="git add requires a non-empty 'paths' list")
        # "--" terminates option parsing: even a path like "-f" stays a path.
        return self._run_git(["add", "--", *[str(p) for p in paths]])

    def _action_commit(self, message: str) -> ToolResult:
        if not isinstance(message, str) or not message.strip():
            return ToolResult(error="git commit requires a non-empty 'message'")
        return self._run_git(["commit", "-m", message])

    def _action_branch(self, branch: str | None = None) -> ToolResult:
        if branch is None:
            return self._run_git(["branch", "--list"])
        if err := self._check_ref(branch, "branch"):
            return ToolResult(error=err)
        return self._run_git(["branch", branch])

    def _action_checkout(self, branch: str, create: bool = False) -> ToolResult:
        if err := self._check_ref(branch, "branch"):
            return ToolResult(error=err)
        argv = ["checkout"] + (["-b"] if create else []) + [branch]
        return self._run_git(argv)

    def _action_push(
        self,
        branch: str,
        remote: str = DEFAULT_REMOTE,
        force: bool = False,
    ) -> ToolResult:
        for value, field in ((remote, "remote"), (branch, "branch")):
            if err := self._check_ref(value, field):
                return ToolResult(error=err)
        prefix, revealed = self._remote_auth()
        argv = prefix + ["push", remote, branch] + (["--force"] if force else [])
        result = self._run_git(argv, revealed)
        # Mark force pushes explicitly so audit entries record the danger
        # (the *input* marking that triggers policy confirmation is the
        # `"force": true` JSON field — see REQUIRE_CONFIRMATION_PATTERNS).
        result.metadata.update(
            {"action": "push", "remote": remote, "branch": branch, "force": bool(force)}
        )
        return result

    def _action_pull(
        self,
        remote: str = DEFAULT_REMOTE,
        branch: str | None = None,
    ) -> ToolResult:
        if err := self._check_ref(remote, "remote"):
            return ToolResult(error=err)
        if branch is not None and (err := self._check_ref(branch, "branch")):
            return ToolResult(error=err)
        prefix, revealed = self._remote_auth()
        argv = prefix + ["pull", remote] + ([branch] if branch else [])
        return self._run_git(argv, revealed)

    def _action_log(self, max_count: int = DEFAULT_LOG_COUNT) -> ToolResult:
        try:
            count = int(max_count)
        except (TypeError, ValueError):
            return ToolResult(error=f"invalid max_count: {max_count!r}")
        count = max(1, min(count, self.MAX_LOG_COUNT))
        return self._run_git(["log", "--oneline", f"--max-count={count}"])

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _check_ref(value: Any, field: str) -> str | None:
        """Reject empty refs and option-injection attempts (``--force`` as a
        branch name would otherwise be parsed as a flag by git)."""
        if not isinstance(value, str) or not value or value.startswith("-"):
            return f"invalid {field}: {value!r}"
        return None

    def _remote_auth(self) -> tuple[list[str], list[str]]:
        """Build the credential argv prefix for remote operations.

        Returns ``(argv_prefix, revealed_secrets)``. The token is resolved
        through the :class:`SecretRegistry` and revealed only here, at the
        point the command line is built; callers pass the revealed value on to
        :meth:`_run_git` so it is scrubbed from any output. Missing token
        yields an empty prefix — git then uses its own credential config.
        """
        token = self._secrets.resolve(self._remote_token_key, required=False)
        if token is None:
            return [], []
        revealed = token.reveal()
        return ["-c", f"http.extraHeader=AUTHORIZATION: bearer {revealed}"], [revealed]

    @staticmethod
    def _subprocess_runner(argv: list[str], cwd: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _run_git(self, argv: list[str], secrets: list[str] | None = None) -> ToolResult:
        """Run ``git <argv>`` and package the outcome as a :class:`ToolResult`.

        All output is scrubbed (known secret patterns plus the literal
        revealed credential) before it can reach the model, the event bus or
        the audit log. The redacted argv is kept in ``metadata`` for audit.
        """
        secrets = secrets or []
        full_argv = ["git", *argv]
        try:
            completed = self._runner(full_argv, self._workdir, self._timeout)
        except FileNotFoundError:
            return ToolResult(error="git executable not found on PATH")
        except subprocess.TimeoutExpired:
            return ToolResult(error=f"git command timed out after {self._timeout:g}s")
        stdout = self._scrub(completed.stdout or "", secrets)
        stderr = self._scrub(completed.stderr or "", secrets)
        metadata = {"argv": [self._scrub(arg, secrets) for arg in full_argv]}
        if completed.returncode != 0:
            return ToolResult(
                error=stderr.strip() or stdout.strip() or f"git exited with code {completed.returncode}",
                metadata=metadata,
            )
        return ToolResult(output=stdout.strip() or "(no output)", metadata=metadata)

    @staticmethod
    def _scrub(text: str, secrets: list[str]) -> str:
        """Remove the literal credential, then apply generic pattern redaction."""
        for secret in secrets:
            if secret:
                text = text.replace(secret, REDACTED)
        return redact_secrets(text)
