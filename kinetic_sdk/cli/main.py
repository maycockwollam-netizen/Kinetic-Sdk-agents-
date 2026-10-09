"""Kinetic CLI: a Claude-Code-style terminal agent (full version)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.ask_user import AskUserTool
from kinetic_sdk.codemap import CodebaseMapTool
from kinetic_sdk.conversation.store import JsonFileConversationStore
from kinetic_sdk.files import (
    ApplyPatchTool,
    FileHistoryTool,
    FileTool,
    GlobTool,
    GrepTool,
)
from kinetic_sdk.git import GitTool
from kinetic_sdk.hooks import HookContext, HookPoint, HookRegistry, HookResult
from kinetic_sdk.llm.client import LiteLLMClient, LLMClient
from kinetic_sdk.memory import InMemoryMemory, JsonFileMemory, MemoryTool
from kinetic_sdk.observability import InMemoryObservabilityLogger
from kinetic_sdk.security.policy import (
    AllowListPolicy,
    PermissionPolicy,
    PermissivePolicy,
)
from kinetic_sdk.terminal import TerminalTool
from kinetic_sdk.testing import MockLLMClient, text_response
from kinetic_sdk.todo import InMemoryTodoStore, TodoReadTool, TodoWriteTool
from kinetic_sdk.workspace import Workspace

CONFIG_DIR = Path.home() / ".kinetic"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROJECT_DIR = Path.cwd() / ".kinetic"
PROJECT_CONFIG = PROJECT_DIR / "project.json"
LAST_RUN_FILE = PROJECT_DIR / "last_run.json"
HISTORY_FILE = PROJECT_DIR / "history.json"

DEFAULT_MODEL = os.environ.get("KINETIC_MODEL", "gpt-4o")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def save_config(key: str, value: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_json(CONFIG_FILE)
    data[key] = value
    CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {key} to {CONFIG_FILE}")


def load_config() -> dict[str, Any]:
    global_config = _load_json(CONFIG_FILE)
    project_config = _load_json(PROJECT_CONFIG)
    return {**global_config, **project_config}


def get_config(key: str, default: Any = None) -> Any:
    return load_config().get(key, os.environ.get(key.upper(), default))


def _confirm_hook_factory(auto_yes: bool):  # type: ignore[no-untyped-def]
    def _hook(ctx: HookContext):  # type: ignore[no-untyped-def]
        if auto_yes:
            return HookResult(should_continue=True)
        try:
            ans = input(
                f"[confirm] tool={ctx.tool_name} input={json.dumps(ctx.tool_input, default=str)[:500]} — allow? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return HookResult(should_continue=False)
        if ans in ("y", "yes"):
            return HookResult(should_continue=True)
        return HookResult(should_continue=False)

    return _hook


def build_agent(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    auto_yes: bool = False,
    safe: bool = False,
    workspace_root: str | Path | None = None,
    stream_logger: bool = False,
) -> Agent:
    model = model or str(get_config("model", DEFAULT_MODEL))
    llm: LLMClient
    api_key = api_key or get_config("api_key", os.environ.get("OPENAI_API_KEY"))
    base_url = base_url or get_config("base_url")

    root = Path(workspace_root) if workspace_root else Path.cwd()
    ws = Workspace(root)
    todo_store = InMemoryTodoStore()
    mem_path = PROJECT_DIR / "memory.json"

    def _ask_handler(question: str) -> str:
        try:
            return input(f"[ask_user] {question}\n> ")
        except (EOFError, KeyboardInterrupt):
            return ""

    tools = [
        TerminalTool(),
        GitTool(),
        FileTool(ws),
        ApplyPatchTool(ws),
        GlobTool(ws),
        GrepTool(ws),
        FileHistoryTool(ws),
        CodebaseMapTool(str(root)),
        MemoryTool(InMemoryMemory()),
        TodoWriteTool(todo_store),
        TodoReadTool(todo_store),
        AskUserTool(handler=_ask_handler),
    ]

    if mem_path.exists():
        try:
            tools[8] = MemoryTool(JsonFileMemory(str(mem_path)))
        except Exception:
            pass

    if api_key:
        try:
            import litellm  # noqa: F401

            llm = LiteLLMClient(model=model, api_key=api_key, api_base=base_url, max_tokens=1024, timeout=60, max_retries=2)
        except Exception as exc:
            print(f"LiteLLM not available ({exc}), falling back to mock.")
            llm = MockLLMClient([text_response("Mock: pretend I ran the task.")])
    else:
        print("No API key found. Running with mock LLM — set api_key via `kinetic config set api_key ...`.")
        llm = MockLLMClient([text_response("Mock: pretend I ran the task.")])

    if safe:
        policy: PermissionPolicy = AllowListPolicy(
            always_allow=["ask_user", "todo_read", "todo_write", "glob", "grep", "codebase_map", "file_history"],
            require_confirmation_patterns={
                "terminal": TerminalTool.REQUIRE_CONFIRMATION_PATTERNS,
                "git": GitTool.REQUIRE_CONFIRMATION_PATTERNS,
            },
        )
    else:
        policy = PermissivePolicy()

    hooks = HookRegistry()
    hooks.register(HookPoint.ON_PERMISSION_CHECK, _confirm_hook_factory(auto_yes))

    obs = InMemoryObservabilityLogger() if stream_logger else None
    store: JsonFileConversationStore | None = None
    if PROJECT_DIR.exists():
        try:
            store = JsonFileConversationStore(str(HISTORY_FILE))
        except Exception:
            store = None

    return Agent(
        llm=llm,
        tools=tools,
        permission_policy=policy,
        hooks=hooks,
        observability_logger=obs,
        state_store=store,
        tool_timeout=120.0,
    )


def _save_evidence(agent: Agent, task: str, result: str) -> None:
    ev = agent.verification_evidence
    payload: dict[str, Any] = {
        "task": task,
        "result": result,
        "run_id": agent.run_id,
        "usage": agent.usage.snapshot().__dict__ if hasattr(agent.usage, "snapshot") else {},
    }
    if ev is not None:
        payload["evidence"] = {
            "changed_files": ev.changed_files,
            "commands_run": ev.commands_run,
            "test_results": ev.test_results,
            "remaining_risks": ev.remaining_risks,
            "verified": ev.is_verified(),
        }
    else:
        payload["evidence"] = None
    try:
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        LAST_RUN_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"Warning: could not persist last run: {exc}", file=sys.stderr)


def print_evidence(agent: Agent) -> None:
    ev = agent.verification_evidence
    if ev is None:
        print("Evidence: none")
        return
    print("Evidence:")
    print(f"  changed_files: {ev.changed_files}")
    print(f"  commands_run: {ev.commands_run}")
    print(f"  test_results: {ev.test_results}")
    print(f"  remaining_risks: {ev.remaining_risks}")
    print(f"  verified: {ev.is_verified()}")


def _stream_into_agent(agent: Agent, task: str, verify: bool) -> str:
    chunks: list[str] = []

    def _on_event(event):  # type: ignore[no-untyped-def]
        if event.event_type == "agent.text_delta":
            delta = event.payload.get("delta", "")
            print(delta, end="", flush=True)
            chunks.append(str(delta))

    agent.event_bus.subscribe("agent.text_delta", _on_event)
    try:
        result = agent.run(task, verify=verify, stream=True)
    finally:
        agent.event_bus.unsubscribe("agent.text_delta", _on_event)
    if chunks:
        print()
    return result


def cmd_run(task: str, model: str | None, api_key: str | None, base_url: str | None, verify: bool, stream: bool, auto_yes: bool, safe: bool) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe)
    try:
        result = _stream_into_agent(agent, task, verify) if stream else agent.run(task, verify=verify)
        print(result)
        print_evidence(agent)
        _save_evidence(agent, task, result)
        try:
            print(f"Usage: {agent.usage.snapshot()}")
        except Exception:
            pass
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_chat(model: str | None, api_key: str | None, base_url: str | None, verify: bool, auto_yes: bool, safe: bool, stream: bool) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe)
    print("Kinetic interactive chat. Type /help for commands, /exit to quit.")
    while True:
        try:
            line = input("kinetic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            print("Bye!")
            return 0
        if line in ("/help", "/h"):
            print("Commands: /exit, /reset, /verify, /evidence, /config, /model <name>, /usage")
            continue
        if line == "/reset":
            agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe)
            print("Conversation reset.")
            continue
        if line == "/verify":
            print(f"verification_required = {agent.verification_required}")
            continue
        if line == "/evidence":
            print_evidence(agent)
            continue
        if line == "/usage":
            try:
                print(agent.usage.snapshot())
            except Exception as exc:
                print(f"Usage unavailable: {exc}")
            continue
        if line == "/config":
            for k, v in load_config().items():
                print(f"  {k}: {v}")
            continue
        if line.startswith("/model "):
            model = line.split(maxsplit=1)[1].strip()
            agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe)
            print(f"Switched to model: {model}")
            continue
        try:
            result = _stream_into_agent(agent, line, verify) if stream else agent.run(line, verify=verify)
            print(result)
            print_evidence(agent)
            _save_evidence(agent, line, result)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.set:
        if "=" not in args.set:
            print("Usage: kinetic config --set key=value")
            return 1
        key, value = args.set.split("=", 1)
        save_config(key, value)
        return 0
    if args.get:
        print(get_config(args.get))
        return 0
    for k, v in load_config().items():
        print(f"{k}: {v}")
    return 0


def cmd_init() -> int:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    if PROJECT_CONFIG.exists():
        print("Project already initialized.")
        return 0
    PROJECT_CONFIG.write_text(json.dumps({
        "model": DEFAULT_MODEL,
        "api_key": "",
        "base_url": "",
        "theme": "default"
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Created {PROJECT_CONFIG}")
    return 0


def cmd_bench(model: str | None, profile: str | None, runs: int) -> int:
    env = os.environ.copy()
    env["MODEL"] = model or str(get_config("model", "mock"))
    env["PROFILE"] = profile or str(get_config("profile", "mock"))
    env["RUNS"] = str(runs)
    script = Path.cwd() / "scripts" / "run_baseline.sh"
    if not script.exists():
        print("Không tìm thấy scripts/run_baseline.sh — bạn đang ở đúng repo Kinetic chứ?")
        return 1
    cmd = ["bash", str(script)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


def cmd_verify() -> int:
    data = _load_json(LAST_RUN_FILE)
    if not data:
        print(f"No last run found at {LAST_RUN_FILE}. Run `kinetic run ...` first.")
        print("Note: `verify` trong chat vẫn dùng /evidence cho run hiện tại.")
        return 1
    print(f"Task: {data.get('task')}")
    print(f"Run: {data.get('run_id')}")
    ev = data.get("evidence")
    if ev is None:
        print("Evidence: none")
    else:
        print("Evidence:")
        for k in ("changed_files", "commands_run", "test_results", "remaining_risks", "verified"):
            print(f"  {k}: {ev.get(k)}")
    print(f"Usage: {data.get('usage')}")
    return 0


def cmd_version() -> int:
    from importlib.metadata import PackageNotFoundError, version
    try:
        v = version("kinetic-agent-sdk")
    except PackageNotFoundError:
        v = "dev"
    print(v)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="kinetic", description="Kinetic AI coding agent CLI")
    sub = parser.add_subparsers(dest="command")

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model")
        p.add_argument("--api-key")
        p.add_argument("--base-url")
        p.add_argument("--verify", action="store_true", help="Require verification evidence")
        p.add_argument("--stream", action="store_true", help="Stream text deltas live")
        p.add_argument("--yes", action="store_true", help="Auto-confirm permission prompts")
        p.add_argument("--safe", action="store_true", help="Deny-by-default policy with confirmations")

    p_run = sub.add_parser("run", help="Run a one-shot task")
    p_run.add_argument("task", help="Task description")
    _add_common(p_run)

    p_chat = sub.add_parser("chat", help="Interactive chat session")
    _add_common(p_chat)

    p_config = sub.add_parser("config", help="Manage configuration")
    p_config.add_argument("--set", metavar="KEY=VALUE")
    p_config.add_argument("--get", metavar="KEY")

    sub.add_parser("init", help="Create .kinetic/project.json")

    p_bench = sub.add_parser("bench", help="Run benchmark tasks")
    p_bench.add_argument("--model")
    p_bench.add_argument("--profile")
    p_bench.add_argument("--runs", type=int, default=1)

    sub.add_parser("verify", help="Show verification evidence from last run")
    sub.add_parser("version", help="Show version")

    args, unknown = parser.parse_known_args()

    if args.command == "run":
        return cmd_run(args.task, args.model, args.api_key, args.base_url, args.verify, args.stream, args.yes, args.safe)
    if args.command == "chat":
        return cmd_chat(args.model, args.api_key, args.base_url, args.verify, args.yes, args.safe, args.stream)
    if args.command == "config":
        return cmd_config(args)
    if args.command == "init":
        return cmd_init()
    if args.command == "bench":
        return cmd_bench(args.model, args.profile, args.runs)
    if args.command == "verify":
        return cmd_verify()
    if args.command == "version":
        return cmd_version()

    if unknown and not unknown[0].startswith("-"):
        task = " ".join(unknown)
        return cmd_run(task, None, None, None, False, False, False, False)

    return cmd_chat(None, None, None, False, False, False, False)


if __name__ == "__main__":
    raise SystemExit(main())
