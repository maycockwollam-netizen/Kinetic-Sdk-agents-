"""Versioned, atomic on-disk cache for static codebase maps."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from kinetic_sdk.codemap.builder import DEFAULT_EXCLUDE, build_codebase_map
from kinetic_sdk.codemap.models import CodebaseMap
from kinetic_sdk.workspace.manager import LocalWorkspace

CODEMAP_SCHEMA_VERSION = 1


class CodebaseMapCache:
    """Disk cache for one root's CodebaseMap, invalidated by Python file changes."""

    def __init__(self, root_path: str, cache_path: str | os.PathLike[str] | None = None) -> None:
        self._workspace = LocalWorkspace(root_path)
        self.path = Path(cache_path) if cache_path is not None else Path(self._workspace.root_path) / ".kinetic" / "codemap-cache.json"

    def _fingerprints(self) -> dict[str, list[int]]:
        root = Path(self._workspace.root_path)
        values: dict[str, list[int]] = {}
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in DEFAULT_EXCLUDE]
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = Path(directory) / filename
                    try:
                        stat = file_path.stat()
                    except OSError:
                        continue
                    values[file_path.relative_to(root).as_posix()] = [stat.st_mtime_ns, stat.st_size]
        return values

    def _load(self) -> tuple[CodebaseMap, dict[str, list[int]]] | None:
        if not self.path.exists():
            return None
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != CODEMAP_SCHEMA_VERSION:
                return None
            map_data, fingerprints = payload.get("codebase_map"), payload.get("fingerprints")
            if not isinstance(map_data, dict) or not isinstance(fingerprints, dict):
                return None
            if not all(isinstance(key, str) and isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value) for key, value in fingerprints.items()):
                return None
            return CodebaseMap.from_dict(map_data), fingerprints
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _save(self, codebase_map: CodebaseMap, fingerprints: dict[str, list[int]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": CODEMAP_SCHEMA_VERSION, "codebase_map": codebase_map.to_dict(), "fingerprints": fingerprints}, handle, ensure_ascii=False)
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def get_or_build(self, *, force_rebuild: bool = False) -> CodebaseMap:
        """Return a valid cached map, or rebuild all files and atomically persist it."""
        fingerprints = self._fingerprints()
        cached = None if force_rebuild else self._load()
        if cached is not None and cached[1] == fingerprints:
            return cached[0]
        codebase_map = build_codebase_map(self._workspace.root_path)
        self._save(codebase_map, fingerprints)
        return codebase_map
