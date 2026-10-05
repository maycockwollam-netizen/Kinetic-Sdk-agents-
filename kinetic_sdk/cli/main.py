"""Kinetic CLI: a Claude-Code-style terminal agent."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.git import GitTool
from kinetic_sdk.llm.client import LiteLLMClient
from kinetic_sdk.memory import InMemoryMemory, MemoryTool
from kinetic_sdk.security.policy import PermissivePolicy
from kinetic_sdk.terminal import TerminalTool
from kinetic_sdk.testing import MockLLMClient, text_response
from kinetic_sdk.todo import InMemoryTodoStore, TodoReadTool, TodoWriteTool

CONFIG_DIR = Path.home() / ".kinetic"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROJECT_DIR = Path.cwd() / ".kinetic"
PROJECT_CONFIG = PROJECT_DIR / "project.json"

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
    # project config overrides global for repo-specific settings
    return {**global_config, **project_config}


def get_config(key: str, default: Any = None) -> Any:
    return load_config().get(key, os.environ.get(key.upper(), default))


def build_agent(model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> Agent:
    model = model or get_config("model", DEFAULT_MODEL)
    api_key = api_key or get_config("api_key", os.environ.get("OPENAI_API_KEY"))
    base_url = base_url or get_config("base_url")

    store = InMemoryTodoStore()
    tools = [
        TerminalTool(),
        GitTool(),
        MemoryTool(InMemoryMemory()),
        TodoWriteTool(store),
        TodoReadTool(store),
    ]

    if api_key:
        try:
            import litellm  # noqa: F401
            llm = LiteLLMClient(model=model, api_key=api_key, api_base=base_url, max_tokens=1024, timeout=60, max_retries=2)
        except Exception as exc:
            print(f"LiteLLM not available ({exc}), falling back to mock.")
            llm = MockLLMClient([text_response("Mock: pretend I ran the task.")])
    else:
        print("No API key found. Running with mock LLM — set KINETIC_API_KEY or `kinetic config set api_key ...` for real runs.")
        llm = MockLLMClient([text_response("Mock: pretend I ran the task.")])

    return Agent(
        llm=llm,
        tools=tools,
        permission_policy=PermissivePolicy(),
    )


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


def cmd_run(task: str, model: str | None, api_key: str | None, base_url: str | None, verify: bool) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url)
    try:
        result = agent.run(task, verify=verify)
        print(result)
        print_evidence(agent)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_chat(model: str | None, api_key: str | None, base_url: str | None, verify: bool) -> int:
    agent = build_agent(model=model, api_key=api_key, base_url=base_url)
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
            print("Commands: /exit, /reset, /verify, /evidence, /config, /model <name>")
            continue
        if line == "/reset":
            agent = build_agent(model=model, api_key=api_key, base_url=base_url)
            print("Conversation reset.")
            continue
        if line == "/verify":
            print(f"verification_required = {agent.verification_required}")
            continue
        if line == "/evidence":
            print_evidence(agent)
            continue
        if line == "/config":
            for k, v in load_config().items():
                print(f"  {k}: {v}")
            continue
        if line.startswith("/model "):
            model = line.split(maxsplit=1)[1].strip()
            agent = build_agent(model=model, api_key=api_key, base_url=base_url)
            print(f"Switched to model: {model}")
            continue
        # treat as a task
        try:
            result = agent.run(line, verify=verify)
            print(result)
            print_evidence(agent)
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
    # show all
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
    env["MODEL"] = model or get_config("model", "mock")
    env["PROFILE"] = profile or get_config("profile", "mock")
    env["RUNS"] = str(runs)
    script = Path.cwd() / "scripts" / "run_baseline.sh"
    if not script.exists():
        print("Không tìm thấy scripts/run_baseline.sh — bạn đang ở đúng repo Kinetic chứ?")
        return 1
    cmd = ["bash", str(script)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


def cmd_verify(agent: Agent) -> int:
    print_evidence(agent)
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

    p_run = sub.add_parser("run", help="Run a one-shot task")
    p_run.add_argument("task", help="Task description")
    p_run.add_argument("--model")
    p_run.add_argument("--api-key")
    p_run.add_argument("--base-url")
    p_run.add_argument("--verify", action="store_true", help="Require verification evidence")

    p_chat = sub.add_parser("chat", help="Interactive chat session")
    p_chat.add_argument("--model")
    p_chat.add_argument("--api-key")
    p_chat.add_argument("--base-url")
    p_chat.add_argument("--verify", action="store_true")

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

    # default: if no subcommand, treat first arg as task or start chat
    args, unknown = parser.parse_known_args()

    if args.command == "run":
        return cmd_run(args.task, args.model, args.api_key, args.base_url, args.verify)
    if args.command == "chat":
        return cmd_chat(args.model, args.api_key, args.base_url, args.verify)
    if args.command == "config":
        return cmd_config(args)
    if args.command == "init":
        return cmd_init()
    if args.command == "bench":
        return cmd_bench(args.model, args.profile, args.runs)
    if args.command == "verify":
        # Need to persist agent? We can't; just print note.
        print("Verify is only meaningful within a chat session (/evidence).")
        return 0
    if args.command == "version":
        return cmd_version()

    # no subcommand: if unknown args look like a task, run it
    if unknown and not unknown[0].startswith("-"):
        task = " ".join(unknown)
        return cmd_run(task, None, None, None, False)

    # otherwise interactive
    return cmd_chat(None, None, None, False)


if __name__ == "__main__":
    raise SystemExit(main())
