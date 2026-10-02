"""Strict, dependency-free benchmark task definitions and loader."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class BenchBudget:
    max_llm_calls: int | None = None
    max_total_tokens: int | None = None
    max_tool_calls: int | None = None
    max_wall_seconds: float | None = None

@dataclass(frozen=True)
class BenchTask:
    id: str
    title: str
    category: str
    difficulty: str
    prompt: str
    fixture_dir: Path
    hidden_tests_dir: Path | None = None
    allowed_tools: list[str] = field(default_factory=list)
    policy_profile: str = "default"
    budget: BenchBudget = field(default_factory=BenchBudget)
    graders: list[dict[str, Any]] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

def _toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
        with path.open("rb") as f: return tomllib.load(f)
    except ModuleNotFoundError:  # Python 3.10 stdlib fallback for our flat schema
        result: dict[str, Any] = {}; section = result
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line: continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1]; section = result.setdefault(name, {})
                if not isinstance(section, dict): raise ValueError(f"task.toml:{number}: invalid section {name}")
                continue
            if "=" not in line: raise ValueError(f"task.toml:{number}: expected key = value")
            key, value = (x.strip() for x in line.split("=", 1))
            if value.startswith('"') and value.endswith('"'): parsed: Any = value[1:-1]
            elif value in ("true", "false"): parsed = value == "true"
            elif value.startswith("[") and value.endswith("]"): parsed = [x.strip().strip('"') for x in value[1:-1].split(",") if x.strip()]
            else:
                try: parsed = int(value)
                except ValueError:
                    try: parsed = float(value)
                    except ValueError: raise ValueError(f"task.toml:{number}: unsupported TOML value")
            section[key] = parsed
        return result

def _need(data: dict[str, Any], key: str, typ: type) -> Any:
    value = data.get(key)
    if not isinstance(value, typ) or (typ is str and isinstance(value, str) and not value.strip()):
        raise ValueError(f"task.toml: field '{key}' must be a non-empty {typ.__name__}")
    return value

def _paths(root: Path, value: str, field_name: str, required: bool = True) -> Path | None:
    path = (root / value).resolve()
    if os.path.commonpath((str(root.resolve()), str(path))) != str(root.resolve()):
        raise ValueError(f"task.toml: field '{field_name}' must stay inside task directory")
    if required and not path.is_dir(): raise ValueError(f"task.toml: field '{field_name}' is not a directory: {value}")
    return path

def load_task(task_dir: str | Path) -> BenchTask:
    root = Path(task_dir); path = root / "task.toml"
    if not path.is_file(): raise ValueError(f"missing task.toml in {root}")
    data = _toml(path)
    allowed = {"id","title","category","difficulty","prompt","fixture_dir","hidden_tests_dir","allowed_tools","policy_profile","budget","graders","risk_tags"}
    unknown = set(data) - allowed
    if unknown: raise ValueError(f"task.toml: unknown field(s): {', '.join(sorted(unknown))}")
    for key in ("id","title","category","difficulty","prompt","fixture_dir"):_need(data,key,str)
    for key in ("allowed_tools","risk_tags"):
        if key in data and (not isinstance(data[key], list) or not all(isinstance(x,str) for x in data[key])): raise ValueError(f"task.toml: field '{key}' must be an array of strings")
    budget_data=data.get("budget", {})
    if not isinstance(budget_data,dict): raise ValueError("task.toml: field 'budget' must be a table")
    unknown_budget=set(budget_data)-{"max_llm_calls","max_total_tokens","max_tool_calls","max_wall_seconds"}
    if unknown_budget: raise ValueError(f"task.toml: unknown budget field(s): {', '.join(sorted(unknown_budget))}")
    for key, value in budget_data.items():
        if not isinstance(value,(int,float)) or isinstance(value,bool) or value <= 0: raise ValueError(f"task.toml: field 'budget.{key}' must be positive")
    raw_graders=data.get("graders", [])
    if isinstance(raw_graders,dict): raw_graders=[{"type": name, **config} for name, config in raw_graders.items()]
    if not isinstance(raw_graders,list) or not all(isinstance(x,dict) for x in raw_graders): raise ValueError("task.toml: field 'graders' must be an array/table of grader tables")
    grader_fields = {"tests": {"type", "command", "expected_exit_code", "timeout"}, "command": {"type", "command", "expected_exit_code", "timeout"}, "diff_constraints": {"type", "allowed_files", "forbidden_files", "max_files", "max_lines"}, "no_secret_leak": {"type", "canary"}, "policy_denied": {"type", "tool_name", "denied"}}
    for grader in raw_graders:
        kind = grader.get("type")
        if kind not in grader_fields: raise ValueError(f"task.toml: unknown grader type {kind!r}")
        extra = set(grader) - grader_fields[kind]
        if extra: raise ValueError(f"task.toml: unknown field(s) in {kind} grader: {', '.join(sorted(extra))}")
    hidden=data.get("hidden_tests_dir")
    if hidden is not None and not isinstance(hidden,str): raise ValueError("task.toml: field 'hidden_tests_dir' must be a string")
    return BenchTask(_need(data,"id",str),_need(data,"title",str),_need(data,"category",str),_need(data,"difficulty",str),_need(data,"prompt",str),cast(Path, _paths(root,data["fixture_dir"],"fixture_dir")),_paths(root,hidden,"hidden_tests_dir") if hidden else None,list(data.get("allowed_tools",[])),data.get("policy_profile","default"),BenchBudget(**budget_data),list(raw_graders),list(data.get("risk_tags",[])))

def load_tasks(root: str | Path) -> list[BenchTask]:
    return [load_task(p) for p in sorted(Path(root).iterdir()) if p.is_dir() and (p / "task.toml").is_file()]
