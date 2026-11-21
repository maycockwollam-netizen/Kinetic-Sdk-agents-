"""Auto-generated ``repo`` skills: explore a checkout and write a SKILL.md.

OpenHands explores a repo by prompting the agent to dig around every time.
This module goes one step further: it scans a checkout read-only, extracts a
compact orientation (overview, layout, build/test commands, conventions), and
writes it as a SKILL.md so a later agent can get up to speed immediately
without re-scanning the tree.
"""

from __future__ import annotations

import re
from pathlib import Path

from kinetic_sdk.skills.skill import (
    MAX_NAME_LENGTH,
    SKILL_FILE_NAME,
    SKILL_NAME_PATTERN,
    Skill,
)

DEFAULT_EXCLUDES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".tox",
    ".mypy_cache", ".pytest_cache", ".kinetic", "dist", "build", "target",
    "htmlcov", ".coverage",
}

_README_NAMES = ("README.md", "README.rst", "README.txt")
_AGENTS_NAMES = ("AGENTS.md",)
_MANIFEST_NAMES = (
    "pyproject.toml", "setup.py", "setup.cfg", "package.json", "Cargo.toml",
    "go.mod", "Makefile", "makefile", "tox.ini", "justfile", "pom.xml",
)
_TEST_DIR_NAMES = ("tests", "test", "specs")


def _root_name(root: Path) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")
    return name[:MAX_NAME_LENGTH] or "repo"


def _read_preview(path: Path, lines: int = 6) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    preview = [line.strip() for line in text.splitlines()[: lines + 8] if line.strip()]
    preview = [line for line in preview if not line.startswith(("#", "<!--"))]
    return "\n".join(preview[:lines]).strip() or None


def _top_level(root: Path) -> list[str]:
    entries: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name in DEFAULT_EXCLUDES or child.name.startswith("."):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    return entries


def _test_dirs(root: Path) -> list[str]:
    return [name for name in _TEST_DIR_NAMES if (root / name).is_dir()]


def _build_test_hints(root: Path) -> list[str]:
    hints: list[str] = []
    for name in _MANIFEST_NAMES:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if name == "pyproject.toml":
            if "pytest" in text:
                hints.append("`python -m pytest` (pytest configured in pyproject.toml)")
            if "maturin" in text or ("rust" in text.lower() and "cargo" in text.lower()):
                hints.append("`maturin build --release` (Rust extension present)")
        elif name == "package.json":
            hints.append("`npm install` / `npm test`")
        elif name == "Cargo.toml":
            hints.append("`cargo build` / `cargo test`")
        elif name == "Makefile":
            hints.append("`make` targets (Makefile present)")
        elif name == "go.mod":
            hints.append("`go build ./...` / `go test ./...`")
        elif name == "tox.ini":
            hints.append("`tox` (tox.ini present)")
        elif name == "justfile":
            hints.append("`just` (justfile present)")
    test_dirs = _test_dirs(root)
    if test_dirs:
        hints.append("Tests live in: " + ", ".join(test_dirs))
    return hints


def _render_skill_md(name: str, description: str, body: str) -> str:
    return (
        f"---\nname: {name}\ndescription: {description}\ntype: repo\n---\n\n"
        f"{body.rstrip()}\n"
    )


def build_repo_skill(
    root_path: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    out_root: str | Path | None = None,
) -> Skill:
    """Scan *root_path* and write a ``repo`` skill for it.

    The generated ``<name>/SKILL.md`` captures the overview (README), layout,
    build/test hints and ANY AGENTS.md conventions, so a later agent can
    orient itself without re-walking the tree. The skill is written under
    ``out_root`` (default: ``<root>/.kinetic/skills``) and returned as a
    validated :class:`Skill` with ``source="generated"``.

    Raises:
        ValueError: ``root_path`` is not an existing directory, or ``name``
            (when given) is not a valid skill name.
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    skill_name = name or _root_name(root)
    if not SKILL_NAME_PATTERN.match(skill_name) or len(skill_name) > MAX_NAME_LENGTH:
        raise ValueError(
            f"invalid skill name {skill_name!r}: must match "
            f"{SKILL_NAME_PATTERN.pattern!r} and be <= {MAX_NAME_LENGTH} chars"
        )

    readme = next((_read_preview(root / n) for n in _README_NAMES if (root / n).is_file()), None)
    agents = next((_read_preview(root / n, lines=10) for n in _AGENTS_NAMES if (root / n).is_file()), None)
    if description is None:
        description = readme or f"Orientation guide for the {root.name} repository."

    sections: list[str] = [f"# {skill_name}", ""]
    if readme:
        sections += ["## Overview", "", readme, ""]
    structure = _top_level(root)
    if structure:
        sections += ["## Repository layout", ""]
        sections += [f"- `{entry}`" for entry in structure]
        sections += [""]
    hints = _build_test_hints(root)
    sections += ["## Build & test", ""]
    if hints:
        sections += [f"- {hint}" for hint in hints]
    else:
        sections += ["No obvious build/test manifest was detected; check the repo for setup docs."]
    sections += [""]
    sections += ["## Conventions", ""]
    sections += [agents or "No AGENTS.md conventions file found."]
    body = "\n".join(sections)

    out_dir = (Path(out_root) if out_root is not None else root / ".kinetic" / "skills") / skill_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SKILL_FILE_NAME).write_text(
        _render_skill_md(skill_name, description, body), encoding="utf-8"
    )
    return Skill.from_directory(out_dir, source="generated")