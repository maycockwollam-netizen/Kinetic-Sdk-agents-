"""Persistence for :class:`ConversationState` — save/resume across processes.

The in-memory state dies with the process; a store keeps it. Wire one into an
agent via ``Agent(state_store=...)``: the agent loads the saved state at
construction (unless an explicit ``state`` was passed) and persists after
every turn, so a crash mid-run loses at most the in-flight turn. Resuming a
compacted conversation resumes with the compacted history — that is the
correct semantics (the elided span is already gone).

Security note: the saved file contains the RAW conversation, which may embed
credentials from tool outputs. Stores do not redact — redaction would corrupt
the history the model needs. Treat the file like a secret (permissions,
encryption at rest) the same way you treat a shell history file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kinetic_sdk.conversation.state import ConversationState

logger = logging.getLogger(__name__)

#: Current on-disk schema version. Loaders must reject anything newer.
SCHEMA_VERSION = 1


class ConversationStoreError(Exception):
    """Raised when a stored conversation cannot be read or parsed."""


class ConversationStore(ABC):
    """Interface for conversation persistence backends."""

    @abstractmethod
    def save(self, state: ConversationState) -> None:
        """Persist *state*, replacing any previously saved one."""

    @abstractmethod
    def load(self) -> ConversationState | None:
        """Return the saved state, or ``None`` when nothing was saved yet.

        Implementations must raise :class:`ConversationStoreError` on a
        corrupted or unreadable existing save — silently starting fresh would
        lose history without anyone noticing.
        """

    @abstractmethod
    def delete(self) -> None:
        """Remove any saved state. No-op when nothing was saved."""


class JsonFileConversationStore(ConversationStore):
    """Single-file JSON store with atomic writes (tmp file + rename).

    One file holds one conversation. Writes go to a temp file in the same
    directory and are renamed over the target, so a crash mid-write never
    leaves a half-written file behind.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: ConversationState) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": state_to_dict(state),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self) -> ConversationState | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversationStoreError(
                f"Cannot load conversation from {self.path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or "state" not in payload:
            raise ConversationStoreError(
                f"Cannot load conversation from {self.path}: not a conversation file"
            )
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ConversationStoreError(
                f"Cannot load conversation from {self.path}: schema version "
                f"{version!r} is not supported (expected {SCHEMA_VERSION})"
            )
        try:
            return state_from_dict(payload["state"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ConversationStoreError(
                f"Cannot load conversation from {self.path}: {exc}"
            ) from exc

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def state_to_dict(state: ConversationState) -> dict[str, Any]:
    """Serialise a state to a plain JSON-compatible dict."""
    return {
        "system_prompt": state.system_prompt,
        "messages": state.messages,
        "max_messages": state.max_messages,
        "metadata": state.metadata,
    }


def state_from_dict(data: dict[str, Any]) -> ConversationState:
    """Rebuild a state from :func:`state_to_dict` output, with validation."""
    if not isinstance(data, dict):
        raise TypeError("state payload must be a dict")
    messages = data.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("state.messages must be a list of dicts")
    state = ConversationState(
        system_prompt=data.get("system_prompt"),
        messages=[dict(m) for m in messages],
        max_messages=data.get("max_messages"),
        metadata=dict(data.get("metadata") or {}),
    )
    return state
