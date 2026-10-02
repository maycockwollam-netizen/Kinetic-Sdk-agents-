"""Verification evidence derived exclusively from recorded trace events."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kinetic_sdk.observability.trace import RunTrace
from kinetic_sdk.project.manifest import ProjectManifest
from kinetic_sdk.verify.parsers import (
    FailureSummary,
    parse_mypy,
    parse_pytest,
    parse_ruff,
)


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class CommandRun:
    command: str
    exit_code: int | None
    timestamp: str
    output: str
    parsed: FailureSummary | None = None


@dataclass
class VerificationContract:
    status: VerificationStatus
    changed_files: list[str] = field(default_factory=list)
    commands_run: list[CommandRun] = field(default_factory=list)
    edits_after_last_verification: bool = False
    parsed_results: list[FailureSummary] = field(default_factory=list)
    remaining_risks: list[str] = field(default_factory=list)
    claims_mismatch: list[str] = field(default_factory=list)

    @classmethod
    def from_trace(
        cls, trace: RunTrace, manifest: ProjectManifest | None, final_text: str | None = None
    ) -> "VerificationContract":
        if manifest is None:
            return cls(VerificationStatus.NOT_APPLICABLE, remaining_risks=["không có manifest"])
        changed: list[str] = []
        commands: list[CommandRun] = []
        pending: dict[str, dict[str, Any]] = {}
        last_edit = ""
        last_success = ""
        failures = False
        for event in trace.events:
            payload = event.get("payload", {})
            event_type = event.get("event_type")
            if event_type == "agent.tool_call_started":
                call_id = str(payload.get("id", ""))
                pending[call_id] = payload
                paths = _changed_paths(str(payload.get("name", "")), payload.get("arguments", {}))
                for path in paths:
                    if path not in changed:
                        changed.append(path)
                if paths:
                    last_edit = str(event.get("timestamp", ""))
            elif event_type == "agent.tool_call_finished":
                start = pending.get(str(payload.get("id", "")), {})
                name = str(payload.get("name", start.get("name", "")))
                args = start.get("arguments", {})
                output = str(payload.get("output", payload.get("output_preview", "")))
                if name == "git":
                    for path in _git_paths(output):
                        if path not in changed:
                            changed.append(path)
                if name != "terminal" or not isinstance(args, dict) or not isinstance(args.get("command"), str):
                    continue
                command = args["command"]
                metadata = payload.get("metadata", {})
                exit_code = metadata.get("exit_code") if isinstance(metadata, dict) else None
                exit_code = exit_code if isinstance(exit_code, int) else None
                parsed = _parse_command(command, output)
                run = CommandRun(command, exit_code, str(event.get("timestamp", "")), output, parsed)
                commands.append(run)
                if _is_declared_verification(command, manifest):
                    if exit_code == 0:
                        last_success = run.timestamp
                    else:
                        failures = True
        edits_after = bool(last_edit and (not last_success or last_edit > last_success))
        contract = cls(
            VerificationStatus.UNVERIFIED,
            changed_files=changed,
            commands_run=commands,
            edits_after_last_verification=edits_after,
            parsed_results=[c.parsed for c in commands if c.parsed is not None],
        )
        if manifest.verification.kind == "diff":
            contract.status = _diff_status(trace, manifest)
        elif failures:
            contract.status = VerificationStatus.FAILED
            contract.remaining_risks.append("lệnh kiểm chứng khai trong manifest đã thất bại")
        elif last_success and not edits_after:
            contract.status = VerificationStatus.VERIFIED
        else:
            contract.remaining_risks.append("chưa có lệnh kiểm chứng thành công sau lần sửa cuối")
        contract.claims_mismatch = _claim_mismatches(final_text, contract)
        if contract.claims_mismatch and contract.status is VerificationStatus.VERIFIED:
            contract.status = VerificationStatus.UNVERIFIED
        return contract


def _changed_paths(name: str, arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    if name == "file" and arguments.get("action") in {"write", "edit", "replace", "delete"}:
        path = arguments.get("path")
        return [path] if isinstance(path, str) else []
    if name in {"apply_patch", "patch"}:
        patch = arguments.get("patch")
        if isinstance(patch, str):
            return re.findall(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", patch, re.MULTILINE)
    return []


def _git_paths(output: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", output, re.MULTILINE)


def _parse_command(command: str, output: str) -> FailureSummary | None:
    if "pytest" in command:
        return parse_pytest(output)
    if "ruff" in command:
        return parse_ruff(output)
    if "mypy" in command:
        return parse_mypy(output)
    return None


def _is_declared_verification(command: str, manifest: ProjectManifest) -> bool:
    kind = manifest.verification.kind
    expected = {"tests": manifest.test, "lint": manifest.lint, "custom": manifest.verification.command}.get(kind)
    return expected is not None and command.strip() == expected.strip()


def _diff_status(trace: RunTrace, manifest: ProjectManifest) -> VerificationStatus:
    limit = manifest.max_diff_lines
    if limit is None:
        return VerificationStatus.NOT_APPLICABLE
    git = [e for e in trace.events if e.get("event_type") == "agent.tool_call_finished" and e.get("payload", {}).get("name") == "git"]
    if not git:
        return VerificationStatus.UNVERIFIED
    output = str(git[-1].get("payload", {}).get("output", ""))
    return VerificationStatus.VERIFIED if len(output.splitlines()) <= limit else VerificationStatus.FAILED


def _claim_mismatches(final_text: str | None, contract: VerificationContract) -> list[str]:
    if not final_text:
        return []
    try:
        claimed = json.loads(final_text)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(claimed, dict):
        return []
    mismatches: list[str] = []
    if "changed_files" in claimed and claimed["changed_files"] != contract.changed_files:
        mismatches.append("changed_files trong báo cáo không khớp trace")
    if "commands_run" in claimed:
        actual = [{"command": run.command, "exit_code": run.exit_code} for run in contract.commands_run]
        if claimed["commands_run"] != actual:
            mismatches.append("commands_run trong báo cáo không khớp trace")
    return mismatches
