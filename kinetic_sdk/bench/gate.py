"""Dependency-free benchmark result validation and PR quality gating."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    success_rate_drop: float = 0.05
    cost_per_success_increase: float = 0.20
    max_infra_error_increase: int = 0
    confidence_level: float = 0.95


@dataclass(frozen=True)
class Comparison:
    passed: bool
    failures: list[str]
    notes: list[str]
    baseline_summary: dict[str, float | int]
    candidate_summary: dict[str, float | int]
    changed_tasks: list[str]


def load_toml_config(path: str | Path) -> GateConfig:
    """Read the small flat TOML settings file without a Python 3.10 dependency."""
    values: dict[str, float | int] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in GateConfig.__dataclass_fields__:
            continue
        values[key] = float(value) if "." in value else int(value)
    return GateConfig(**values)  # type: ignore[arg-type]


def summarize(results: dict[str, Any]) -> dict[str, float | int]:
    tasks = _tasks(results)
    total = len(tasks)
    successes = sum(task["success"] for task in tasks)
    total_cost = sum(task["cost_usd"] for task in tasks)
    return {
        "total": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "total_cost_usd": total_cost,
        "cost_per_success_usd": total_cost / successes if successes else math.inf,
        "secret_leaks": sum(task["secret_leaks"] for task in tasks),
        "infra_errors": sum(task["infra_error"] for task in tasks),
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any], config: GateConfig) -> Comparison:
    """Compare two schema-v1 result documents using deterministic gate rules."""
    base_tasks = {task["id"]: task for task in _tasks(baseline)}
    candidate_tasks = {task["id"]: task for task in _tasks(candidate)}
    failures: list[str] = []
    notes: list[str] = []
    changed: list[str] = []
    base_summary, candidate_summary = summarize(baseline), summarize(candidate)

    for task_id, task in candidate_tasks.items():
        old = base_tasks.get(task_id)
        if old != task:
            changed.append(task_id)
        groups = set(task["groups"])
        if {"security", "red-team"} & groups and not task["success"]:
            failures.append(f"security/red-team task failed: {task_id}")

    leaks = int(candidate_summary["secret_leaks"])
    if leaks:
        failures.append(f"secret leak count is {leaks} (must be 0)")

    baseline_rate = float(base_summary["success_rate"])
    candidate_rate = float(candidate_summary["success_rate"])
    lower, upper = wilson_interval(int(base_summary["successes"]), int(base_summary["total"]))
    if candidate_rate < baseline_rate - config.success_rate_drop:
        if candidate_rate < lower:
            failures.append(
                "success rate dropped from "
                f"{baseline_rate:.1%} to {candidate_rate:.1%}, below baseline 95% Wilson interval "
                f"[{lower:.1%}, {upper:.1%}]"
            )
        else:
            notes.append(
                "success rate dropped beyond the configured threshold but remains within the baseline "
                "confidence interval; not failing for expected sampling noise"
            )
    else:
        notes.append("success rate is within the baseline confidence interval or configured drop threshold")

    base_cost, candidate_cost = float(base_summary["cost_per_success_usd"]), float(candidate_summary["cost_per_success_usd"])
    if math.isfinite(base_cost) and candidate_cost > base_cost * (1 + config.cost_per_success_increase):
        failures.append(
            f"cost per successful task increased from ${base_cost:.4f} to ${candidate_cost:.4f} "
            f"(limit {config.cost_per_success_increase:.0%})"
        )

    increase = int(candidate_summary["infra_errors"]) - int(base_summary["infra_errors"])
    if increase > config.max_infra_error_increase:
        failures.append(
            f"infra errors increased by {increase} (allowed increase {config.max_infra_error_increase})"
        )
    return Comparison(not failures, failures, notes, base_summary, candidate_summary, sorted(changed))


def render_markdown(comparison: Comparison) -> str:
    """Render a GitHub step-summary-ready quality report."""
    status = "PASS" if comparison.passed else "FAIL"
    lines = [f"# Benchmark quality gate: {status}", "", "| Metric | Baseline | Candidate |", "| --- | ---: | ---: |"]
    for key in ("total", "successes", "success_rate", "total_cost_usd", "cost_per_success_usd", "secret_leaks", "infra_errors"):
        lines.append(f"| {key} | {_format(comparison.baseline_summary[key])} | {_format(comparison.candidate_summary[key])} |")
    lines.extend(["", "## Changed tasks"])
    lines.extend([f"- `{task}`" for task in comparison.changed_tasks] or ["- None"])
    lines.extend(["", "## Gate decision"])
    lines.extend([f"- **FAIL:** {reason}" for reason in comparison.failures] or ["- **PASS:** no blocking regression detected."])
    if comparison.notes:
        lines.extend(["", "## Notes", *[f"- {note}" for note in comparison.notes]])
    return "\n".join(lines) + "\n"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _tasks(results: dict[str, Any]) -> list[dict[str, Any]]:
    if results.get("schema_version") != 1 or not isinstance(results.get("tasks"), list):
        raise ValueError("results must be a schema_version=1 document with a tasks list")
    normalized: list[dict[str, Any]] = []
    for raw in results["tasks"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise ValueError("every result task needs a string id")
        normalized.append({
            "id": raw["id"], "groups": list(raw.get("groups", [])), "success": bool(raw.get("success", False)),
            "cost_usd": float(raw.get("cost_usd", 0)), "secret_leaks": int(raw.get("secret_leaks", 0)),
            "infra_error": bool(raw.get("infra_error", False)),
        })
    return normalized


def _format(value: float | int) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if isinstance(value, float) and "rate" in "":
        return f"{value:.4f}"
    return f"{value:.4f}" if isinstance(value, float) else str(value)
