"""Exceptions raised by the subagent package.

Every message names the sub-agent, agent id, or limit involved so the
caller (and the audit trail) can see exactly *what* failed — the same
convention the other packages follow.
"""

from __future__ import annotations


class SubagentError(RuntimeError):
    """Base class for sub-agent spawning/execution failures."""


class SubagentSpecError(ValueError):
    """A :class:`~kinetic_sdk.subagent.manifest.SubagentSpec` is invalid.

    Raised for a malformed ``name`` or an empty ``system_prompt``. The
    message names the offending field and value.
    """


class BudgetExceededError(SubagentError):
    """The shared :class:`~kinetic_sdk.subagent.budget.SpawnBudget` is spent.

    Raised the moment any agent in the spawned tree requests one tool call
    beyond ``max_total_tool_calls``. This is a *hard stop* for the agent
    whose turn is in flight — it propagates out of ``Agent.run()`` and is
    converted to an error ``ToolResult`` by the ``DelegateTool`` one level
    up, so a runaway tree is throttled instead of burning API budget.
    """


class RepetitionLimitError(SubagentError):
    """One agent repeated the exact same tool call too many times in a row.

    Raised by :class:`~kinetic_sdk.subagent.budget.RepetitionCircuitBreaker`
    when a single agent calls (same tool, same arguments) more than
    ``max_consecutive_repeats`` times consecutively — the classic signature
    of a stuck agent loop.
    """


class FileLockError(SubagentError):
    """Base class for sub-agent file-coordination failures."""


class FileLockTimeoutError(FileLockError, TimeoutError):
    """A file remained owned by another agent until the wait expired."""


class FileLockOwnershipError(FileLockError):
    """An agent attempted to release or renew a lock it does not own."""
