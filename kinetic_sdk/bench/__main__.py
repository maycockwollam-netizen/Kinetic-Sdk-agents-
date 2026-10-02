"""CLI for repeatable benchmark runs and reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.bench.report import build_report, markdown
from kinetic_sdk.bench.runner import BenchRunner
from kinetic_sdk.bench.task import BenchTask, load_tasks
from kinetic_sdk.security.policy import AllowListPolicy, PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, text_response


def main() -> None:
    parser=argparse.ArgumentParser(prog="python -m kinetic_sdk.bench"); sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run");run.add_argument("--tasks",default="benchmarks/tasks");run.add_argument("--filter");run.add_argument("--runs",type=int,default=3);run.add_argument("--model");run.add_argument("--profile",default="default");run.add_argument("--out",required=True);run.add_argument("--mock-script");run.add_argument("--unsafe-local-exec",action="store_true")
    report=sub.add_parser("report");report.add_argument("results")
    args=parser.parse_args()
    if args.command=="report":
        result=build_report(json.loads(Path(args.results).read_text(encoding="utf-8")));print(markdown(result));return
    tasks=load_tasks(args.tasks)
    if args.filter: tasks=[t for t in tasks if args.filter==t.id or args.filter in t.risk_tags or args.filter==t.category]
    scripts=json.loads(Path(args.mock_script).read_text(encoding="utf-8")) if args.mock_script else {}
    def factory(task:BenchTask,workspace:Path)->Agent:
        entries=[text_response(x) for x in scripts.get(task.id, scripts.get("default", ["mock completed"]))]
        policy=PermissivePolicy() if args.profile in ("dev","development") else AllowListPolicy()
        return Agent(MockLLMClient(entries,model=args.model or "mock-model"),permission_policy=policy)
    BenchRunner(factory,runs=args.runs,unsafe_local_exec=args.unsafe_local_exec,profile=args.profile).run(tasks).write(args.out)
if __name__=="__main__":main()
