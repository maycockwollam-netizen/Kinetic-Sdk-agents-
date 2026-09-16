"""Versioned, atomic persistence for recorded agent replays."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from kinetic_sdk.replay.models import ReplayRun

REPLAY_SCHEMA_VERSION = 1


class ReplayStoreError(Exception):
    """Raised when a replay cannot be read or written safely."""


class ReplayStore(ABC):
    """Persistence interface for one replay run."""

    @abstractmethod
    def save(self, replay: ReplayRun) -> None:
        """Atomically replace the saved replay with *replay*."""

    @abstractmethod
    def load(self) -> ReplayRun | None:
        """Return the replay, or ``None`` when it has not been saved yet."""

    @abstractmethod
    def delete(self) -> None:
        """Remove the replay. No-op when it does not exist."""


class JsonFileReplayStore(ReplayStore):
    """One replay JSON file, written via temp-file plus atomic rename."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, replay: ReplayRun) -> None:
        payload = {"schema_version": REPLAY_SCHEMA_VERSION, "replay": replay.to_dict()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, default=str)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self) -> ReplayRun | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayStoreError(f"Cannot load replay from {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or "replay" not in payload:
            raise ReplayStoreError(f"Cannot load replay from {self.path}: not a replay file")
        if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
            raise ReplayStoreError(
                f"Cannot load replay from {self.path}: schema version "
                f"{payload.get('schema_version')!r} is not supported "
                f"(expected {REPLAY_SCHEMA_VERSION})"
            )
        try:
            replay_data = payload["replay"]
            if not isinstance(replay_data, dict):
                raise ValueError("replay must be a dict")
            return ReplayRun.from_dict(replay_data)
        except (TypeError, ValueError) as exc:
            raise ReplayStoreError(f"Cannot load replay from {self.path}: {exc}") from exc

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
