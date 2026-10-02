"""Defensive parsers for common verification tool output."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Failure:
    id: str
    location: str | None
    message: str
    kind: str


@dataclass(frozen=True)
class FailureSummary:
    failures: list[Failure] = field(default_factory=list)
    unparsed: bool = False
    raw: str = ""

    @property
    def signature(self) -> str:
        value = "\n".join(f"{f.kind}|{f.id}|{f.location}|{f.message}" for f in self.failures)
        return hashlib.sha256(value.encode()).hexdigest()[:16]


def _raw(text: str) -> str:
    return text[:4000]


def parse_pytest(text: str) -> FailureSummary:
    failures: list[Failure] = []
    for line in text.splitlines():
        match = re.match(r"FAILED\s+(\S+)\s+-\s+(.+)", line)
        if match:
            nodeid, message = match.groups()
            failures.append(Failure(nodeid, nodeid.split("::")[0], message[:300], "test"))
    return FailureSummary(failures, not bool(failures) and "failed" in text.lower(), _raw(text))


def parse_ruff(text: str) -> FailureSummary:
    try:
        rows = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return FailureSummary(unparsed=True, raw=_raw(text))
    if not isinstance(rows, list):
        return FailureSummary(unparsed=True, raw=_raw(text))
    failures: list[Failure] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        loc = row.get("location") or {}
        filename, code, message = row.get("filename"), row.get("code"), row.get("message")
        if not isinstance(filename, str) or not isinstance(code, str) or not isinstance(message, str):
            continue
        line = loc.get("row") if isinstance(loc, dict) else None
        failures.append(Failure(code, f"{filename}:{line}" if isinstance(line, int) else filename, message[:300], "ruff"))
    return FailureSummary(failures, False, _raw(text))


def parse_mypy(text: str) -> FailureSummary:
    failures: list[Failure] = []
    for line in text.splitlines():
        match = re.match(r"(.+?):(\d+): error: (.*?)(?:\s+\[([^]]+)\])?$", line)
        if match:
            filename, line_no, message, code = match.groups()
            failures.append(Failure(code or "error", f"{filename}:{line_no}", message[:300], "mypy"))
    return FailureSummary(failures, not bool(failures) and "error:" in text, _raw(text))
