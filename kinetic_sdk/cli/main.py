"""Kinetic CLI: a Claude-Code-style terminal agent (full version)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.ask_user import AskUserTool
from kinetic_sdk.cli import ui as tui
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
from kinetic_sdk.tool.base import Tool, ToolResult
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


def _default_plan_confirm(plan: str) -> bool:
    """Show the proposed plan and ask the user to approve it."""
    tui.print_system("PLAN PROPOSED — approve before the agent acts:")
    print(plan)
    try:
        answer = input("\nApprove plan and proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


class PlanGateTool(Tool):
    """Plan-mode gate: the agent must propose a plan and get approval first.

    When enabled, the model is instructed to call this tool with its plan
    BEFORE making changes. Execution returns an error until the plan is
    approved, so the agent is forced to revise instead of acting.
    """

    name = "plan_gate"
    description = (
        "You are in PLAN mode. Before making ANY change (file edits, git "
        "operations, running commands) propose your complete step-by-step plan "
        "by calling this tool with the 'plan' argument. Do NOT edit files or run "
        "commands until the plan is approved. If it is rejected, revise your plan "
        "and propose it again via plan_gate."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "plan": {"type": "string", "description": "The complete plan to approve."},
        },
        "required": ["plan"],
    }

    def __init__(self, on_propose: Callable[[str], bool]) -> None:
        if not callable(on_propose):
            raise TypeError("on_propose must be callable")
        self._on_propose = on_propose

    def execute(self, plan: str, **_: Any) -> ToolResult:  # type: ignore[override]
        if not isinstance(plan, str) or not plan.strip():
            return ToolResult(error="plan must be a non-empty string")
        if self._on_propose(plan):
            return ToolResult(output="Plan approved. You may proceed.")
        return ToolResult(
            error="Plan rejected. Revise your plan and propose it again via plan_gate."
        )


def build_agent(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    auto_yes: bool = False,
    safe: bool = False,
    workspace_root: str | Path | None = None,
    stream_logger: bool = False,
    session_id: str | None = None,
    plan_mode: bool = False,
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

    if plan_mode:
        tools.append(PlanGateTool(_default_plan_confirm))

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
            if session_id is None:
                store = JsonFileConversationStore(str(HISTORY_FILE))
            else:
                from kinetic_sdk.conversation.sessions import SessionStore

                store = SessionStore(str(PROJECT_DIR / "sessions")).open(session_id)
        except Exception:  # noqa: BLE001 - session persistence is best-effort
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
    tui.print_evidence_panel(agent.verification_evidence)


def _stream_into_agent(agent: Agent, task: str, verify: bool) -> str:
    chunks: list[str] = []

    def _on_event(event):  # type: ignore[no-untyped-def]
        if event.event_type == "agent.text_delta":
            delta = event.payload.get("delta", "")
            tui.stream_delta(str(delta))
            chunks.append(str(delta))

    agent.event_bus.subscribe("agent.text_delta", _on_event)
    try:
        result = agent.run(task, verify=verify, stream=True)
    finally:
        agent.event_bus.unsubscribe("agent.text_delta", _on_event)
    if chunks:
        print()
    return result


def cmd_run(task: str, model: str | None, api_key: str | None, base_url: str | None, verify: bool, stream: bool, auto_yes: bool, safe: bool, show_diff: bool = False, plan: bool = False) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe, plan_mode=plan)
    try:
        if stream:
            result = _stream_into_agent(agent, task, verify)
        else:
            with tui.status("Running"):
                result = agent.run(task, verify=verify)
        tui.print_assistant(result)
        print_evidence(agent)
        _save_evidence(agent, task, result)
        if show_diff:
            cmd_diff(Path.cwd())
        try:
            tui.print_ok(f"Usage: {agent.usage.snapshot()}")
        except Exception:
            pass
        return 0
    except Exception as exc:
        tui.print_error(str(exc))
        return 1


def _tui_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def cmd_session(args: argparse.Namespace) -> int:
    """Manage named conversation sessions (list / show / delete)."""
    from kinetic_sdk.conversation.sessions import SessionStore

    store = SessionStore(str(PROJECT_DIR / "sessions"))
    action = args.session_action
    if action == "list":
        sessions = store.list()
        if not sessions:
            print("No saved sessions yet. Start a chat with `kinetic chat --session <name>`.")
            return 0
        print("SESSION\tMSGS\tUPDATED (UTC)")
        for info in sessions:
            print(f"{info.id}\t{info.message_count}\t{info.updated_at}")
        return 0
    if action == "show":
        state = store.load(args.session_id)
        if state is None:
            tui.print_error(f"session {args.session_id!r} not found")
            return 1
        tui.print_ok(f"Session {args.session_id}: {len(state.messages)} message(s)")
        for message in state.messages:
            role = message.get("role", "?")
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(block.get("text", block.get("name", ""))) for block in content
                )
            preview = str(content)[:160]
            print(f"[{role}] {preview}")
        return 0
    if action == "delete":
        deleted = store.delete(args.session_id)
        if not deleted:
            tui.print_error(f"session {args.session_id!r} not found")
            return 1
        tui.print_ok(f"Deleted session {args.session_id}")
        return 0
    return 0


def cmd_chat(model: str | None, api_key: str | None, base_url: str | None, verify: bool, auto_yes: bool, safe: bool, stream: bool, session_id: str | None = None, plan_mode: bool = False) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe, session_id=session_id, plan_mode=plan_mode)
    if _tui_available():
        from kinetic_sdk.cli.tui_app import KineticTuiApp

        KineticTuiApp(agent).run()
        return 0
    tui.print_banner()
    while True:
        try:
            line = input("kinetic> ").strip()
        except (EOFError, KeyboardInterrupt):
            tui.print_ok("Bye!")
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            tui.print_ok("Bye!")
            return 0
        if line in ("/help", "/h"):
            tui.print_system("Commands: /exit, /reset, /verify, /evidence, /config, /diff, /model <name>, /usage")
            continue
        if line == "/diff":
            cmd_diff(Path.cwd())
            continue
        if line == "/plan":
            status = "ON — the agent must propose a plan before acting" if plan_mode else "OFF"
            tui.print_system(f"plan mode: {status}")
            continue
        if line == "/reset":
            agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe, session_id=session_id, plan_mode=plan_mode)
            tui.print_system("Conversation reset.")
            continue
        if line == "/verify":
            tui.print_system(f"verification_required = {agent.verification_required}")
            continue
        if line == "/evidence":
            print_evidence(agent)
            continue
        if line == "/usage":
            try:
                tui.print_assistant(str(agent.usage.snapshot()))
            except Exception as exc:
                tui.print_error(f"Usage unavailable: {exc}")
            continue
        if line == "/config":
            for k, v in load_config().items():
                tui.print_system(f"{k}: {v}")
            continue
        if line.startswith("/model "):
            model = line.split(maxsplit=1)[1].strip()
            agent = build_agent(model=model, api_key=api_key, base_url=base_url, auto_yes=auto_yes, safe=safe, session_id=session_id, plan_mode=plan_mode)
            tui.print_system(f"Switched to model: {model}")
            continue
        tui.print_user(line)
        try:
            result = _stream_into_agent(agent, line, verify) if stream else agent.run(line, verify=verify)
            tui.print_assistant(result)
            print_evidence(agent)
            _save_evidence(agent, line, result)
        except Exception as exc:
            tui.print_error(str(exc))
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


def _workspace_diff(root: str | Path) -> str:
    """Return the working-tree diff vs HEAD (stat + body), or '' when clean."""
    root_path = Path(root)
    if not (root_path / ".git").is_dir():
        return ""
    try:
        stat = subprocess.run(
            ["git", "-C", str(root_path), "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        body = subprocess.run(
            ["git", "-C", str(root_path), "diff", "HEAD", "--color", "never"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (stat.stdout + "\n" + body.stdout).strip()


def cmd_diff(path: str | Path = ".") -> int:
    """Show working-tree changes vs HEAD for *path* (default: current dir)."""
    diff = _workspace_diff(path)
    if not diff:
        tui.print_system("No working-tree changes (or not a git repository).")
        return 0
    tui.print_system(f"Working-tree changes in {Path(path).resolve()}:")
    print(diff)
    return 0


def _doctor_tick(ok: bool, label: str, detail: str = "") -> None:
    mark = "✓" if ok else "✗"
    suffix = f" — {detail}" if detail else ""
    print(f"{mark} {label}{suffix}")


def cmd_doctor() -> int:
    """Check the environment/runtime and print a pass/fail health report."""
    import importlib.metadata
    import importlib.util
    import platform

    _doctor_tick(
        sys.version_info >= (3, 10),
        f"Python {platform.python_version()} (needs >=3.10)",
    )
    try:
        installed = importlib.metadata.version("kinetic-agent-sdk")
    except importlib.metadata.PackageNotFoundError:
        installed = "n/a (source checkout)"
    _doctor_tick(True, f"kinetic-agent-sdk {installed}")

    try:
        from kinetic_sdk import _kinetic_rs  # type: ignore[attr-defined]

        tokens = _kinetic_rs.count_tokens("xin chào", "cl100k_base")
        _doctor_tick(True, "Rust extension _kinetic_rs", f"count_tokens OK ({tokens})")
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        _doctor_tick(
            False,
            "Rust extension _kinetic_rs",
            f"unavailable ({type(exc).__name__}) — reinstall with the Rust toolchain",
        )

    for package in ("litellm", "rich", "textual", "tiktoken"):
        found = importlib.util.find_spec(package) is not None
        _doctor_tick(found, f"{package}", "installed" if found else "not installed (optional)")

    is_git = (Path.cwd() / ".git").is_dir() or (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    _doctor_tick(is_git, "git repository", "cwd is a git work tree" if is_git else "not a git work tree")

    env_key = next(
        (
            name
            for name in (
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "GROQ_API_KEY", "OPENHANDS_API_KEY",
            )
            if os.environ.get(name)
        ),
        None,
    )
    _doctor_tick(env_key is not None, "LLM API key", env_key or "none set (use --api-key or config)")

    cargo = importlib.util.find_spec("maturin") is not None
    _doctor_tick(cargo, "maturin", "installed (dev/build)" if cargo else "not installed")
    return 0


def cmd_explore(path: str = ".") -> int:
    """Scan *path* and write a ``repo`` skill (SKILL.md) for it."""
    from kinetic_sdk.codemap.skill import build_repo_skill

    try:
        skill = build_repo_skill(path)
    except ValueError as exc:
        tui.print_error(str(exc))
        return 1
    tui.print_ok(f"Wrote repo skill '{skill.name}' to {skill.root_path}")
    tui.print_system(skill.description)
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
    p_run.add_argument("--diff", action="store_true", help="Show working-tree diff after running")
    p_run.add_argument("--plan", action="store_true", help="Plan mode: agent must propose a plan before acting")
    _add_common(p_run)

    p_chat = sub.add_parser("chat", help="Interactive chat session")
    p_chat.add_argument("--session", metavar="ID", help="Named session to resume or create")
    p_chat.add_argument("--plan", action="store_true", help="Plan mode: agent must propose a plan before acting")
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
    sub.add_parser("doctor", help="Check environment/runtime health")
    sub.add_parser("version", help="Show version")
    p_explore = sub.add_parser("explore", help="Auto-generate a repo skill (SKILL.md) for a project")
    p_explore.add_argument("path", nargs="?", default=".")
    p_diff = sub.add_parser("diff", help="Show working-tree changes vs HEAD")
    p_diff.add_argument("path", nargs="?", default=".")
    p_session = sub.add_parser("session", help="Manage conversation sessions")
    sess_sub = p_session.add_subparsers(dest="session_action")
    sess_sub.add_parser("list", help="List saved sessions")
    _sess_show = sess_sub.add_parser("show", help="Show a session's messages")
    _sess_show.add_argument("session_id")
    _sess_delete = sess_sub.add_parser("delete", help="Delete a session")
    _sess_delete.add_argument("session_id")

    args, unknown = parser.parse_known_args()

    if args.command == "run":
        return cmd_run(args.task, args.model, args.api_key, args.base_url, args.verify, args.stream, args.yes, args.safe, args.diff, args.plan)
    if args.command == "chat":
        return cmd_chat(args.model, args.api_key, args.base_url, args.verify, args.yes, args.safe, args.stream, args.session, args.plan)
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
    if args.command == "explore":
        return cmd_explore(args.path)
    if args.command == "session":
        return cmd_session(args)
    if args.command == "diff":
        return cmd_diff(args.path)
    if args.command == "doctor":
        return cmd_doctor()

    if unknown and not unknown[0].startswith("-"):
        task = " ".join(unknown)
        return cmd_run(task, None, None, None, False, False, False, False)

    return cmd_chat(None, None, None, False, False, False, False)


if __name__ == "__main__":
    raise SystemExit(main())
