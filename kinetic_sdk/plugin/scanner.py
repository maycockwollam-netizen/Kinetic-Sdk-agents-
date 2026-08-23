"""Static AST-based scanner for plugin source code.

**Read this first — the scanner is a TRIPWIRE, not a sandbox.**

A plugin is Python code that runs *in the same process* as Kinetic. Unlike
an MCP server (isolated subprocess — worst case, Kinetic disconnects) or a
skill (inert Markdown — the agent still decides whether to follow it), a
malicious plugin needs no exploit and no social engineering: it can read
``os.environ``, open sockets, or monkey-patch ``kinetic_sdk.security`` to
disable permission checks for every request that comes after it, simply by
executing ordinary Python at import time.

Python has no real in-process sandbox, and a determined attacker can evade
ANY static pattern matching (obfuscation, runtime string decoding,
``getattr`` chains, ...). This scanner therefore exists to:

* catch *accidental* violations and *unsophisticated* malicious attempts,
* raise the cost of an attack and leave findings in the audit log,

and explicitly NOT to prove that "plugin passed scan == plugin is safe".
Only load plugins from sources you would trust with your own credentials.
This limitation is deliberate and documented — not a bug to be "fixed" by
adding more patterns.

The scanner parses source with :mod:`ast` (never regex over raw text —
Python must be parsed grammatically; regex both misses and false-positives)
and resolves names through the module's own import aliases, so
``from subprocess import run as r; r(...)`` is caught exactly like
``subprocess.run(...)``.

All block/warn policy lives in :func:`scan_plugin` — the loader only reads
``ScanResult.clean`` and never re-derives "what counts as critical" itself,
mirroring the ``vet_skill``/``SkillRegistry`` split in the skills package.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from importlib import machinery
from pathlib import Path

from kinetic_sdk.plugin.exceptions import PluginLoadError
from kinetic_sdk.plugin.manifest import PluginManifest
from kinetic_sdk.skills.vet import VetFlag

logger = logging.getLogger(__name__)

#: Modules whose import suggests direct network access outside the managed
#: ``mcp/`` channel (warning only — legitimate for plugins that declare it).
NETWORK_MODULES: frozenset[str] = frozenset(
    {"socket", "urllib", "http", "requests", "httpx", "aiohttp"}
)

#: Builtins that execute code from data. Flagged when the input is not a
#: constant literal.
_DYNAMIC_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile"})

#: subprocess entry points that must never be called directly — plugins
#: needing a subprocess go through GitTool (or a future controlled wrapper).
_SUBPROCESS_CALLS: frozenset[str] = frozenset(
    {"Popen", "run", "call", "check_call", "check_output"}
)

#: Direct shell-out through os.
_OS_SHELL_CALLS: frozenset[str] = frozenset({"os.system", "os.popen"})

#: Namespace that must never be *written to* by a plugin. Reading public
#: API (``from kinetic_sdk.security import AllowListPolicy``) is fine;
#: assigning/setattr-ing into it is the clearest signal of an attempt to
#: disable the SDK's safety rails.
_SECURITY_NAMESPACE = "kinetic_sdk.security"


@dataclass(frozen=True)
class ScanResult:
    """Outcome of statically scanning one plugin.

    Attributes:
        clean: ``True`` when no flag has ``severity="critical"``. This is
            the ONLY field the loader consults. ``True`` means "no tripwire
            fired", NOT "this plugin is safe" — see the module docstring.
        flags: Every finding, warnings and criticals alike, reusing
            :class:`~kinetic_sdk.skills.vet.VetFlag` (``location`` holds
            ``path:lineno``).
        files_scanned: Source files that were parsed, for the audit trail.
    """

    clean: bool
    flags: list[VetFlag] = field(default_factory=list)
    files_scanned: list[str] = field(default_factory=list)


class _PluginVisitor(ast.NodeVisitor):
    """AST visitor collecting suspicious-pattern flags for one file.

    ``_aliases`` maps local names to the absolute dotted path they refer to
    (built from ``import`` / ``from ... import`` statements), so calls and
    attribute accesses are compared against canonical paths regardless of
    how the plugin chose to import them.
    """

    def __init__(self, location: str) -> None:
        self.flags: list[VetFlag] = []
        self._location = location
        self._aliases: dict[str, str] = {}

    # --- helpers -----------------------------------------------------------

    def _add(self, severity: str, category: str, message: str, node: ast.AST) -> None:
        self.flags.append(
            VetFlag(
                severity=severity,  # type: ignore[arg-type]
                category=category,
                message=message,
                location=f"{self._location}:{getattr(node, 'lineno', 0)}",
            )
        )

    def _resolve(self, node: ast.AST) -> str | None:
        """Resolve a Name/Attribute chain to a canonical dotted path.

        Returns ``None`` for anything else (calls, subscripts, ...), which
        the checks treat as "unknown" — a tripwire accepts that gap rather
        than pretending to full dataflow analysis.
        """
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        parts.reverse()
        first = self._aliases.get(parts[0], parts[0])
        return ".".join([first, *parts[1:]])

    def _is_security_namespace_write(self, target: ast.AST) -> bool:
        resolved = self._resolve(target)
        return resolved is not None and (
            resolved == _SECURITY_NAMESPACE
            or resolved.startswith(_SECURITY_NAMESPACE + ".")
        )

    # --- imports -----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in NETWORK_MODULES:
                self._add(
                    "warning",
                    "direct_network",
                    f"imports {alias.name!r} — direct network access outside the "
                    "managed mcp/ channel",
                    node,
                )
            if alias.asname:
                self._aliases[alias.asname] = alias.name
            else:
                self._aliases[root] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0] if module else ""
        if root in NETWORK_MODULES:
            self._add(
                "warning",
                "direct_network",
                f"imports from {module!r} — direct network access outside the "
                "managed mcp/ channel",
                node,
            )
        if module:
            for alias in node.names:
                self._aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # --- attribute access / writes -----------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._resolve(node) == "os.environ":
            self._add(
                "critical",
                "env_access",
                "accesses os.environ directly — secrets must go through "
                "SecretRegistry like every other SDK module",
                node,
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Catches `from os import environ` — usage of the bound alias.
        if self._aliases.get(node.id) == "os.environ":
            self._add(
                "critical",
                "env_access",
                "uses os.environ (via from-import) — secrets must go through "
                "SecretRegistry like every other SDK module",
                node,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_write_targets(node.targets, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_write_targets([node.target], node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_write_targets([node.target], node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._check_write_targets(node.targets, node, verb="deletes")
        self.generic_visit(node)

    def _check_write_targets(
        self, targets: list[ast.expr], node: ast.AST, verb: str = "assigns into"
    ) -> None:
        for target in targets:
            for attribute in self._outermost_attributes(target):
                if self._is_security_namespace_write(attribute):
                    self._add(
                        "critical",
                        "security_tamper",
                        f"{verb} the kinetic_sdk.security namespace — writing to "
                        "the SDK's security module can disable permission checks "
                        "for every later request",
                        node,
                    )

    @staticmethod
    def _outermost_attributes(node: ast.AST) -> list[ast.Attribute]:
        """Attribute targets of a write, without their own descendants.

        ``a.b.c = x`` yields only ``a.b.c`` (not ``a.b``), so one write
        produces exactly one flag.
        """
        if isinstance(node, ast.Attribute):
            return [node]
        if isinstance(node, ast.Subscript):
            return _PluginVisitor._outermost_attributes(node.value)
        if isinstance(node, ast.Starred):
            return _PluginVisitor._outermost_attributes(node.value)
        if isinstance(node, (ast.Tuple, ast.List)):
            result: list[ast.Attribute] = []
            for element in node.elts:
                result.extend(_PluginVisitor._outermost_attributes(element))
            return result
        return []

    # --- calls ---------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        resolved = self._resolve(node.func)

        if resolved in _DYNAMIC_BUILTINS:
            inputs = list(node.args) + [kw.value for kw in node.keywords]
            if any(not isinstance(value, ast.Constant) for value in inputs) or not inputs:
                self._add(
                    "critical",
                    "code_execution",
                    f"calls {resolved}() with non-literal input — dynamic code "
                    "execution evades static review",
                    node,
                )
        elif resolved == "__import__" or resolved == "importlib.import_module":
            if not node.args or not (
                isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                self._add(
                    "critical",
                    "dynamic_import",
                    f"calls {resolved}() with a non-constant module name — "
                    "dynamic imports evade static analysis",
                    node,
                )
        elif resolved in _OS_SHELL_CALLS:
            self._add(
                "critical",
                "process_spawn",
                f"calls {resolved}() directly — subprocesses must go through a "
                "controlled SDK wrapper (e.g. GitTool), not raw shell calls",
                node,
            )
        elif resolved is not None and resolved.startswith("subprocess."):
            if resolved.rsplit(".", 1)[-1] in _SUBPROCESS_CALLS:
                self._add(
                    "critical",
                    "process_spawn",
                    f"calls {resolved}() directly — subprocesses must go through "
                    "a controlled SDK wrapper (e.g. GitTool), not raw subprocess",
                    node,
                )
        elif resolved == "setattr":
            if node.args and self._is_security_namespace_write(node.args[0]):
                self._add(
                    "critical",
                    "security_tamper",
                    "setattr() targeting the kinetic_sdk.security namespace — "
                    "monkey-patching the SDK's security module can disable "
                    "permission checks for every later request",
                    node,
                )
        elif resolved == "getattr":
            # getattr(os, "environ") — the classic attribute-evasion form.
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "environ"
                and self._resolve(node.args[0]) == "os"
            ):
                self._add(
                    "critical",
                    "env_access",
                    "getattr(os, 'environ') — secrets must go through "
                    "SecretRegistry like every other SDK module",
                    node,
                )
        elif resolved == "open":
            if node.args and not (
                isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                self._add(
                    "warning",
                    "dynamic_file_access",
                    "open() with a non-literal path — may read/write outside the "
                    "workspace",
                    node,
                )
        self.generic_visit(node)


class StaticPluginScanner:
    """Parses plugin source with :mod:`ast` and collects findings.

    The class holds no state between files; one instance can scan many
    plugins. Findings reuse :class:`~kinetic_sdk.skills.vet.VetFlag` so the
    audit/log formatting code paths are shared with skill vetting.
    """

    def scan_source(self, source: str, location: str) -> list[VetFlag]:
        """Scan one source string. Unparsable code is itself critical."""
        try:
            tree = ast.parse(source, filename=location)
        except SyntaxError as exc:
            return [
                VetFlag(
                    severity="critical",
                    category="unparseable",
                    message=f"source cannot be parsed as Python ({exc.msg})",
                    location=f"{location}:{exc.lineno or 0}",
                )
            ]
        visitor = _PluginVisitor(location)
        visitor.visit(tree)
        return visitor.flags


def _find_spec_no_import(dotted: str):
    """Locate a module spec WITHOUT importing any of its parent packages.

    ``importlib.util.find_spec`` imports parent packages for dotted names —
    which would execute plugin code *before* the scan. Walking
    :class:`~importlib.machinery.PathFinder` level by level avoids that.
    """
    parts = dotted.split(".")
    spec = machinery.PathFinder.find_spec(parts[0], None)
    walked = parts[0]
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return None
        walked = f"{walked}.{part}"
        spec = machinery.PathFinder.find_spec(walked, spec.submodule_search_locations)
    return spec


def _locate_source_files(manifest: PluginManifest) -> list[Path]:
    """Return the Python files that make up *manifest*'s plugin.

    Directory plugins: every ``*.py`` under the plugin root (the whole
    plugin is scanned, not just the entry module). Entry-point plugins: the
    entry module's file, or every ``*.py`` under it when it is a package —
    located WITHOUT importing anything (see :func:`_find_spec_no_import`).
    """
    if manifest.source == "directory":
        root = Path(manifest.directory or "")
        files = sorted(root.rglob("*.py"))
        if not files:
            raise PluginLoadError(
                f"plugin {manifest.name!r}: no Python files found under {root}"
            )
        return files
    spec = _find_spec_no_import(manifest.module_spec)
    if spec is None or spec.origin is None or not os.path.isfile(spec.origin):
        raise PluginLoadError(
            f"plugin {manifest.name!r}: cannot locate module "
            f"{manifest.module_spec!r} on sys.path without importing it"
        )
    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        return sorted(origin.parent.rglob("*.py"))
    return [origin]


def scan_plugin(
    manifest: PluginManifest, scanner: StaticPluginScanner | None = None
) -> ScanResult:
    """Scan one plugin's source BEFORE any import — the single policy point.

    ``ScanResult.clean`` (no critical flags) is the only field the loader
    reads. A critical flag means "blocked"; a warning means "loaded but
    surfaced to the audit log".

    Raises:
        PluginLoadError: the plugin's source files cannot be located or
            read at all (nothing to scan — failing closed would be wrong
            too: there is simply nothing loadable).
    """
    scanner = scanner if scanner is not None else StaticPluginScanner()
    files = _locate_source_files(manifest)
    flags: list[VetFlag] = []
    scanned: list[str] = []
    for path in files:
        scanned.append(str(path))
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PluginLoadError(
                f"plugin {manifest.name!r}: cannot read source file {path}: {exc}"
            ) from exc
        flags.extend(scanner.scan_source(source, str(path)))
    clean = not any(flag.severity == "critical" for flag in flags)
    return ScanResult(clean=clean, flags=flags, files_scanned=scanned)
