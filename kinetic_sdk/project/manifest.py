"""Load ``.kinetic/project.toml`` without guessing project commands."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


class ManifestError(ValueError):
    """The project manifest is malformed or contains unsupported keys."""


@dataclass(frozen=True)
class VerificationConfig:
    kind: Literal["tests", "lint", "diff", "custom"]
    command: str | None = None


@dataclass(frozen=True)
class ProjectManifest:
    """The opt-in verification policy declared by a project."""

    setup: str | None = None
    test: str | None = None
    test_targeted: str | None = None
    lint: str | None = None
    typecheck: str | None = None
    protected_paths: list[str] = field(default_factory=list)
    command_timeout: int | None = None
    max_diff_lines: int | None = None
    verification: VerificationConfig = field(default_factory=lambda: VerificationConfig("tests"))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ProjectManifest":
        try:
            raw: Any
            try:
                import tomllib
            except ImportError:  # Python 3.10: this narrow manifest grammar is stdlib-only.
                raw = _parse_toml_subset(Path(path).read_text(encoding="utf-8"))
            else:
                with open(path, "rb") as stream:
                    raw = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            raise ManifestError(f"cannot read project manifest: {exc}") from exc
        if not isinstance(raw, dict):
            raise ManifestError("manifest root must be a table")
        allowed = {"setup", "test", "test_targeted", "lint", "typecheck", "protected_paths", "command_timeout", "max_diff_lines", "verification"}
        unknown = set(raw) - allowed
        if unknown:
            raise ManifestError(f"unknown manifest key(s): {', '.join(sorted(unknown))}")
        strings = {"setup", "test", "test_targeted", "lint", "typecheck"}
        for key in strings:
            if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
                raise ManifestError(f"{key} must be a non-empty string")
        protected = raw.get("protected_paths", [])
        if not isinstance(protected, list) or not all(isinstance(item, str) and item for item in protected):
            raise ManifestError("protected_paths must be a list of non-empty strings")
        for key in ("command_timeout", "max_diff_lines"):
            if key in raw and (not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] <= 0):
                raise ManifestError(f"{key} must be a positive integer")
        verification = raw.get("verification")
        if not isinstance(verification, dict):
            raise ManifestError("verification must be a table")
        unknown_verification = set(verification) - {"kind", "command"}
        if unknown_verification:
            raise ManifestError(f"unknown verification key(s): {', '.join(sorted(unknown_verification))}")
        kind = verification.get("kind")
        if kind not in {"tests", "lint", "diff", "custom"}:
            raise ManifestError("verification.kind must be tests, lint, diff, or custom")
        command = verification.get("command")
        if command is not None and (not isinstance(command, str) or not command.strip()):
            raise ManifestError("verification.command must be a non-empty string")
        if kind == "custom" and command is None:
            raise ManifestError("verification.command is required for kind=custom")
        return cls(**{key: raw.get(key) for key in strings}, protected_paths=list(protected), command_timeout=raw.get("command_timeout"), max_diff_lines=raw.get("max_diff_lines"), verification=VerificationConfig(kind=kind, command=command))


@dataclass(frozen=True)
class ManifestLoadResult:
    status: Literal["loaded", "no_manifest"]
    manifest: ProjectManifest | None
    path: Path


def load_project_manifest(root: str | os.PathLike[str] = ".") -> ProjectManifest | ManifestLoadResult:
    """Load the manifest or return ``no_manifest``; never invent commands."""
    path = Path(root) / ".kinetic" / "project.toml"
    if not path.is_file():
        return ManifestLoadResult("no_manifest", None, path)
    return ProjectManifest.from_file(path)


def _parse_toml_subset(text: str) -> dict[str, Any]:
    """Parse the documented flat TOML subset on Python 3.10 without deps."""
    result: dict[str, Any] = {}
    table = result
    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if name != "verification":
                raise ManifestError(f"unsupported TOML table at line {number}")
            table = result.setdefault(name, {})
            continue
        if "=" not in line:
            raise ManifestError(f"invalid TOML at line {number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            raise ManifestError(f"invalid TOML assignment at line {number}")
        table[key] = _parse_toml_value(value, number)
    return result


def _parse_toml_value(value: str, number: int) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        values = value[1:-1].strip()
        return [] if not values else [_parse_toml_value(item.strip(), number) for item in values.split(",")]
    if value.isdigit():
        return int(value)
    if value in {"true", "false"}:
        return value == "true"
    raise ManifestError(f"unsupported TOML value at line {number}")
