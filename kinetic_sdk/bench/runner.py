"""Repeatable coding benchmark runner with isolated fixture copies."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, cast

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.budget import RunBudget
from kinetic_sdk.bench.grader import Grade, make_grader
from kinetic_sdk.bench.task import BenchTask
from kinetic_sdk.hooks.base import Hook, HookContext, HookPoint, HookResult
from kinetic_sdk.observability.logger import InMemoryObservabilityLogger
from kinetic_sdk.observability.trace import RunTrace

AgentFactory = Callable[[BenchTask, Path], Agent]
def _snapshot(root:Path)->dict[str,str]: return {p.relative_to(root).as_posix():p.read_text(encoding="utf-8",errors="replace") for p in root.rglob("*") if p.is_file()}
def _sha() -> str | None:
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None
@dataclass
class RunRecord:
    task_id:str; run_index:int; status:str; output:str=""; error:str|None=None; grades:list[dict[str,Any]]=field(default_factory=list); trace:dict[str,Any]=field(default_factory=dict); wall_seconds:float=0; seed:int|None=None; model:str|None=None; profile:str|None=None; git_sha:str|None=None
    def to_dict(self)->dict[str,Any]:return asdict(self)
@dataclass
class BenchResult:
    records:list[RunRecord]
    def to_dict(self)->dict[str,Any]:return {"schema_version":1,"records":[r.to_dict() for r in self.records]}
    def write(self,path:str|Path)->None:Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
class BenchRunner:
    def __init__(self,agent_factory:AgentFactory,*,runs:int=3,unsafe_local_exec:bool=False,seed:int|None=None,profile:str|None=None)->None:
        if runs<1:raise ValueError("runs must be >= 1")
        self.agent_factory,self.runs,self.unsafe_local_exec,self.seed,self.profile=agent_factory,runs,unsafe_local_exec,seed,profile
    def run(self,tasks:Iterable[BenchTask])->BenchResult:return BenchResult([self._run_one(t,i) for t in tasks for i in range(self.runs)])
    def _run_one(self,task:BenchTask,index:int)->RunRecord:
        started=time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"kinetic-bench-{task.id}-") as temporary:
            workspace=Path(temporary)/"workspace"; shutil.copytree(task.fixture_dir,workspace); baseline=_snapshot(workspace)
            obs=InMemoryObservabilityLogger(); output=""; trace=RunTrace("",[])
            try:
                agent=self.agent_factory(task,workspace)
                agent.run_budget=RunBudget(task.budget.max_llm_calls,task.budget.max_total_tokens)
                obs.attach(agent.event_bus)
                tool_calls=0
                if task.budget.max_tool_calls is not None:
                    from kinetic_sdk.hooks.registry import HookRegistry
                    hooks=agent.hooks or HookRegistry(); agent.hooks=hooks
                    max_tool_calls = task.budget.max_tool_calls
                    def cap(ctx: HookContext) -> HookResult | None:
                        nonlocal tool_calls
                        del ctx
                        tool_calls += 1
                        return HookResult(should_continue=False) if tool_calls > max_tool_calls else None
                    hooks.register(HookPoint.BEFORE_TOOL_CALL, cast(Hook, cap))
                output=agent.run(task.prompt)
                trace=RunTrace.collect(obs.entries,agent.run_id or "")
                elapsed=time.monotonic()-started
                if task.budget.max_wall_seconds is not None and elapsed>task.budget.max_wall_seconds:
                    return self._record(task,index,"budget_exceeded",output,"wall-clock budget exceeded",[],trace,elapsed,agent)
                if "budget exceeded" in output.lower() or (task.budget.max_tool_calls is not None and tool_calls>task.budget.max_tool_calls):
                    return self._record(task,index,"budget_exceeded",output,None,[],trace,elapsed,agent)
            except Exception as exc:
                return self._record(task,index,"infra_error","",f"{type(exc).__name__}: {exc}",[],trace,time.monotonic()-started,None)
            # This copy is deliberately after agent.run: hidden tests never enter its view.
            if task.hidden_tests_dir is not None: shutil.copytree(task.hidden_tests_dir,workspace/task.hidden_tests_dir.name)
            try: grades=[make_grader(c).grade(workspace,output,trace,getattr(agent.audit_logger,"entries",[]),baseline) for c in task.graders]
            except Exception as exc:return self._record(task,index,"infra_error",output,f"grader: {type(exc).__name__}: {exc}",[],trace,time.monotonic()-started,agent)
            return self._record(task,index,"pass" if all(g.passed for g in grades) else "agent_fail",output,None,grades,trace,time.monotonic()-started,agent)
    def _record(self,task:BenchTask,index:int,status:str,output:str,error:str|None,grades:list[Grade],trace:RunTrace,elapsed:float,agent:Agent|None)->RunRecord:
        return RunRecord(task.id,index,status,output,error,[asdict(g) for g in grades],trace.to_summary(),round(elapsed,6),self.seed,getattr(getattr(agent,"llm",None),"model",None),self.profile,_sha())
