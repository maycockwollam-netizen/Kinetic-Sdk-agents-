"""Aggregate benchmark JSON records into machine and human reports."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def wilson(passes:int,total:int)->list[float]:
    if not total:return [0.0,0.0]
    z=1.96;p=passes/total;d=1+z*z/total;c=(p+z*z/(2*total))/d;m=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d;return [round(c-m,4),round(c+m,4)]
def build_report(data:dict[str,Any])->dict[str,Any]:
    records=data.get("records",[]); passed=[r for r in records if r["status"]=="pass"]; by: dict[str, list[dict[str, Any]]] = {}
    for r in records:by.setdefault(r["task_id"],[]).append(r)
    tasks={k:{"passed":sum(x["status"]=="pass" for x in v),"runs":len(v)} for k,v in by.items()}
    costs=[r.get("trace",{}).get("usage",{}).get("cost_usd",0) for r in passed]
    return {"total_runs":len(records),"passed":len(passed),"success_rate":len(passed)/len(records) if records else 0,"wilson_95":wilson(len(passed),len(records)),"per_task":tasks,"cost_per_success":sum(costs)/len(passed) if passed else None,"median_wall_seconds":statistics.median([r.get("wall_seconds",0) for r in records]) if records else 0,"median_tool_calls":statistics.median([r.get("trace",{}).get("tool_call_count",0) for r in records]) if records else 0,"regression_rate":0.0,"safety":{"policy_denied_correct":sum(g["passed"] for r in records for g in r.get("grades",[]) if g["name"]=="policy_denied"),"secret_leaks":sum(not g["passed"] for r in records for g in r.get("grades",[]) if g["name"]=="no_secret_leak")}}
def markdown(report:dict[str,Any])->str:
    rows=["# Benchmark report","",f"Success rate: **{report['passed']}/{report['total_runs']}** ({report['success_rate']:.1%}), Wilson 95% {report['wilson_95']}","","| Task | Pass | Runs |","|---|---:|---:|"]
    rows += [f"| {name} | {v['passed']} | {v['runs']} |" for name,v in report['per_task'].items()];return "\n".join(rows)+"\n"
def report_file(path:str|Path)->tuple[dict[str,Any],str]:return build_report(json.loads(Path(path).read_text(encoding="utf-8"))), ""
