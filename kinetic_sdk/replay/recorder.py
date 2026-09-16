"""Event-bus subscriber that durably records a run for later playback."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import state_to_dict
from kinetic_sdk.event.bus import Event, EventBus
from kinetic_sdk.replay.models import ReplayRun, ReplayStep
from kinetic_sdk.replay.store import ReplayStore
from kinetic_sdk.security.redact import redact_value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReplayRecorder:
    """Record redacted events and replay-valid snapshots into a ``ReplayStore``.

    By default snapshots are redacted too. Set ``capture_raw_snapshots=True``
    only for protected local storage when the replay will be used to fork a
    conversation, because raw tool output can contain credentials.

    Reasoning traces are omitted by default because they may contain sensitive
    model output. Set ``capture_reasoning=True`` to include their redacted
    event payloads in the replay.
    """

    def __init__(
        self,
        store: ReplayStore,
        *,
        capture_raw_snapshots: bool = False,
        capture_reasoning: bool = False,
        parent_run_id: str | None = None,
        fork_sequence: int | None = None,
    ) -> None:
        if (parent_run_id is None) != (fork_sequence is None):
            raise ValueError("parent_run_id and fork_sequence must be supplied together")
        self.store = store
        self.capture_raw_snapshots = capture_raw_snapshots
        self.capture_reasoning = capture_reasoning
        self.parent_run_id = parent_run_id
        self.fork_sequence = fork_sequence
        self._run: ReplayRun | None = None
        self._lock = threading.Lock()

    def attach(self, bus: EventBus) -> None:
        """Subscribe to every event emitted on *bus*."""
        bus.subscribe("*", self.handle)

    def detach(self, bus: EventBus) -> None:
        """Stop recording events from *bus*."""
        bus.unsubscribe("*", self.handle)

    def handle(self, event: Event) -> None:
        """Record an event belonging to an agent run; unrelated events skip."""
        if event.type == "agent.reasoning_trace" and not self.capture_reasoning:
            return
        run_id = event.payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        with self._lock:
            if self._run is None or self._run.run_id != run_id:
                self._run = self._new_run(run_id)
            self._append(event.type, redact_value(dict(event.payload)))

    def snapshot(self, state: ConversationState, run_id: str | None) -> None:
        """Record a state snapshot at a provider-valid persistence boundary."""
        if run_id is None:
            return
        state_data = state_to_dict(state)
        payload = {"state": state_data if self.capture_raw_snapshots else redact_value(state_data)}
        with self._lock:
            if self._run is None or self._run.run_id != run_id:
                self._run = self._new_run(run_id)
            self._append("replay.snapshot", payload)

    def _append(self, event_type: str, payload: dict[str, object]) -> None:
        assert self._run is not None
        self._run.steps.append(
            ReplayStep(len(self._run.steps), _utcnow_iso(), event_type, payload)
        )
        self.store.save(self._run)

    def _new_run(self, run_id: str) -> ReplayRun:
        return ReplayRun(
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            fork_sequence=self.fork_sequence,
        )
