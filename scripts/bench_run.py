#!/usr/bin/env python3
"""Run deterministic cassette smoke benchmarks without calling an LLM provider."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.agent.classifier import Classification, TaskClassifier, TaskComplexity
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.security.policy import AllowListPolicy, PermissivePolicy
from kinetic_sdk.testing import MockLLMClient, MockTool, text_response, tool_response


class _ProfileClassifier(TaskClassifier):
    """Offline classifier that pins a benchmark profile without provider I/O."""

    def __init__(self, mode: AgentMode) -> None:
        self.mode = mode

    def classify(self, task: str) -> Classification:
        complexity = TaskComplexity.SIMPLE if self.mode is AgentMode.FLASH else TaskComplexity.COMPLEX
        return Classification(complexity, self.mode, 1.0, "benchmark profile")


def _response(spec: dict[str, Any]):
    if "tool_call" in spec:
        call = spec["tool_call"]
        return tool_response(call.get("id", "cassette-call"), call["name"], call.get("arguments", {}))
    return text_response(str(spec.get("text", "")))


def run_cassette(path: Path, profile: AgentMode) -> dict[str, Any]:
    cassette = json.loads(path.read_text(encoding="utf-8"))
    tools = [MockTool(name="echo", result="echoed")] if cassette.get("uses_echo") else []
    policy = PermissivePolicy() if tools else AllowListPolicy()
    agent = Agent(
        llm=MockLLMClient([_response(item) for item in cassette["responses"]]),
        tools=tools,
        classifier=_ProfileClassifier(profile),
        permission_policy=policy,
    )
    try:
        output = agent.run(cassette["prompt"])
        expected = cassette["expected"]
        leak_markers = cassette.get("secret_canaries", [])
        leaks = sum(marker in output for marker in leak_markers)
        return {
            "id": cassette["id"], "groups": cassette.get("groups", []),
            "success": output == expected and leaks == 0, "cost_usd": 0.0,
            "secret_leaks": leaks, "infra_error": False,
        }
    except Exception:
        return {
            "id": cassette["id"], "groups": cassette.get("groups", []), "success": False,
            "cost_usd": 0.0, "secret_leaks": 0, "infra_error": True,
        }


def run_live_cassette(path: Path, profile: AgentMode, model: str, api_key: str) -> dict[str, Any]:
    """Run a nightly case against a real optional LiteLLM backend."""
    from kinetic_sdk.llm.client import LiteLLMClient

    cassette = json.loads(path.read_text(encoding="utf-8"))
    tools = [MockTool(name="echo", result="echoed")] if cassette.get("uses_echo") else []
    agent = Agent(
        llm=LiteLLMClient(model=model, api_key=api_key), tools=tools,
        classifier=_ProfileClassifier(profile), permission_policy=PermissivePolicy() if tools else AllowListPolicy(),
    )
    try:
        output = agent.run(cassette["prompt"])
        usage = agent.usage.snapshot()
        leaks = sum(marker in output for marker in cassette.get("secret_canaries", []))
        return {"id": cassette["id"], "groups": cassette.get("groups", []),
                "success": output == cassette["expected"] and leaks == 0,
                "cost_usd": usage.cost_usd, "secret_leaks": leaks, "infra_error": False}
    except Exception:
        return {"id": cassette["id"], "groups": cassette.get("groups", []), "success": False,
                "cost_usd": 0.0, "secret_leaks": 0, "infra_error": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-script", type=Path, help="Directory of recorded MockLLM responses")
    parser.add_argument("--model", help="Real LiteLLM model; nightly only")
    parser.add_argument("--api-key", help="Real provider key; nightly only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["FLASH", "MAX"], default="MAX")
    args = parser.parse_args()
    if bool(args.model) != bool(args.api_key) or (args.mock_script is None and args.model is None):
        parser.error("provide --mock-script, or both --model and --api-key")
    script_dir = args.mock_script or Path("benchmarks/cassettes")
    runner = run_cassette if args.mock_script is not None else lambda path, profile: run_live_cassette(path, profile, args.model, args.api_key)
    tasks = [runner(path, AgentMode(args.profile.lower())) for path in sorted(script_dir.glob("*.json"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "mode": "mock", "profile": args.profile, "tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    return 0 if tasks and not any(task["infra_error"] for task in tasks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
