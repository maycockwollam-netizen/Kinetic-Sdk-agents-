"""Durable checkpoints for resuming the same production agent run.

Unlike :class:`ReplayBranch`, a checkpoint is not an experimental branch: it
keeps the original run id and records the small amount of Agent runtime state
that is not part of ``ConversationState``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from kinetic_sdk.conversation.store import state_from_dict
from kinetic_sdk.event.bus import Event
from kinetic_sdk.llm.client import ToolCall
from kinetic_sdk.replay.recorder import ReplayRecorder
from kinetic_sdk.replay.store import ReplayStore
from kinetic_sdk.security.policy import PermissionDecision

if TYPE_CHECKING:
    from kinetic_sdk.agent.agent import Agent
    from kinetic_sdk.llm.client import LLMClient
    from kinetic_sdk.tool.base import Tool


class CheckpointError(RuntimeError):
    """A requested checkpoint is absent, incomplete, or cannot be restored."""


class PendingConfirmationError(RuntimeError):
    """A confirmation awaits a human decision and can be resumed later."""

    def __init__(
        self,
        call: ToolCall,
        arguments: dict[str, Any],
        decision: PermissionDecision,
        checkpoint_id: str,
    ) -> None:
        super().__init__(f"Confirmation pending for tool {call.name!r}")
        self.call = call
        self.arguments = arguments
        self.decision = decision
        self.checkpoint_id = checkpoint_id


class CheckpointManager:
    """Thin replay-backed checkpoint manager for crash/confirmation recovery.

    ``ReplayStore`` has one save slot, so its natural checkpoint identifier is
    the run id. Calling :meth:`save` again for that run replaces its latest
    checkpoint while preserving the replay timeline.
    """

    def __init__(self, store: ReplayStore) -> None:
        self._store = store

    def save(self, agent: "Agent") -> str:
        """Persist raw state plus Agent runtime metadata under its run id."""
        run_id = agent.run_id
        if not run_id:
            raise CheckpointError("Cannot checkpoint an agent before its run has started")
        recorder = ReplayRecorder(self._store, capture_raw_snapshots=True)
        recorder.snapshot(agent.state, run_id)
        recorder.handle(Event("replay.checkpoint", {"run_id": run_id, "agent": agent._checkpoint_runtime_state()}))
        return run_id

    def resume(
        self, checkpoint_id: str, llm: "LLMClient", tools: list["Tool"], **agent_kwargs: Any
    ) -> "Agent":
        """Rebuild an Agent with the original run id and runtime routing state."""
        replay = self._store.load()
        if replay is None or replay.run_id != checkpoint_id:
            raise CheckpointError(f"Checkpoint {checkpoint_id!r} was not found")
        snapshot = next(
            (step for step in reversed(replay.steps) if step.event_type == "replay.snapshot"), None
        )
        metadata = next(
            (step for step in reversed(replay.steps) if step.event_type == "replay.checkpoint"), None
        )
        if snapshot is None or metadata is None:
            raise CheckpointError(f"Checkpoint {checkpoint_id!r} is incomplete")
        state_data = snapshot.payload.get("state")
        runtime = metadata.payload.get("agent")
        if not isinstance(state_data, dict) or not isinstance(runtime, dict):
            raise CheckpointError(f"Checkpoint {checkpoint_id!r} is invalid")
        try:
            state = state_from_dict(copy.deepcopy(state_data))
        except (TypeError, ValueError, KeyError) as exc:
            raise CheckpointError(f"Checkpoint {checkpoint_id!r} has invalid state: {exc}") from exc
        if "state" in agent_kwargs:
            raise ValueError("CheckpointManager owns state; do not pass state=")
        from kinetic_sdk.agent.agent import Agent

        agent = Agent(llm=llm, tools=tools, state=state, **agent_kwargs)
        agent._restore_checkpoint_runtime_state(checkpoint_id, runtime)
        recorder = ReplayRecorder(self._store, capture_raw_snapshots=True)
        recorder.handle(Event("replay.resumed", {"run_id": checkpoint_id, "checkpoint_id": checkpoint_id}))
        return agent


def resume_from_confirmation(
    checkpoint_id: str,
    approved: bool,
    *,
    checkpoint_manager: CheckpointManager,
    llm: "LLMClient",
    tools: list["Tool"],
) -> str:
    """Resolve a pending confirmation then continue its original run."""
    agent = checkpoint_manager.resume(
        checkpoint_id, llm, tools, checkpoint_manager=checkpoint_manager
    )
    agent._resolve_pending_confirmation(approved)
    return agent.run()
