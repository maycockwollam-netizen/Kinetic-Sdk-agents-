"""Named-session management over the single-file JSON conversation stores.

One conversation = one JSON file. This module adds opencode-style multiple
sessions: a :class:`SessionStore` keeps ``<id>.json`` files in one directory
and can list / open / delete them, while each individual session stays a
:class:`~kinetic_sdk.conversation.store.JsonFileConversationStore`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.conversation.store import JsonFileConversationStore

#: Lowercase letters/digits/dashes/underscores; must not be empty.
SESSION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DEFAULT_SESSION = "default"


@dataclass(frozen=True)
class SessionInfo:
    """Lightweight metadata for one saved session."""

    id: str
    path: str
    updated_at: str  # ISO 8601 UTC
    message_count: int


class SessionStore:
    """Multiple named sessions stored as ``<sessions_dir>/<id>.json``."""

    def __init__(self, sessions_dir: str | Path) -> None:
        self.dir = Path(sessions_dir)

    @staticmethod
    def _validate_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "session id must match [a-z0-9][a-z0-9_-]{0,63} (lowercase)"
            )

    def _path(self, session_id: str) -> Path:
        self._validate_id(session_id)
        return self.dir / f"{session_id}.json"

    def list(self) -> list[SessionInfo]:
        """All valid sessions, newest first."""
        if not self.dir.is_dir():
            return []
        infos: list[SessionInfo] = []
        for path in self.dir.glob("*.json"):
            session_id = path.stem
            try:
                self._validate_id(session_id)
            except ValueError:
                continue
            infos.append(self._info(session_id, path))
        return sorted(infos, key=lambda info: info.updated_at, reverse=True)

    def _info(self, session_id: str, path: Path) -> SessionInfo:
        updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        count = 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = payload.get("state", {}).get("messages", [])
            count = len(messages) if isinstance(messages, list) else 0
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        return SessionInfo(session_id, str(path), updated, count)

    def load(self, session_id: str) -> ConversationState | None:
        """Load *session_id*'s saved state, or ``None`` if it has none."""
        path = self._path(session_id)
        if not path.exists():
            return None
        return JsonFileConversationStore(path).load()

    def open(self, session_id: str) -> JsonFileConversationStore:
        """Return a writable store for *session_id* (file created on save)."""
        return JsonFileConversationStore(self._path(session_id))

    def delete(self, session_id: str) -> bool:
        """Delete *session_id*; returns whether it existed."""
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True