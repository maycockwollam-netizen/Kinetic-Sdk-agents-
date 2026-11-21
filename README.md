# Kinetic Agent SDK

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular Python SDK and command-line agent for building tool-calling AI applications. Kinetic supports sync and async agents, provider-neutral LLM clients, secure tool permissions, memory, MCP, subagents, observability, and workspace operations.

## Features

- **Flexible agents** — synchronous and asynchronous agent loops with streaming
- **Tool calling** — terminal, files, git, memory, todo, codebase mapping, MCP, and custom tools
- **Safe by default** — deny-by-default permissions, confirmation hooks, and secret redaction
- **Provider-neutral** — use LiteLLM or implement the `LLMClient` interface
- **Production-ready primitives** — retries, timeouts, context compaction, usage tracking, and OpenTelemetry
- **Extensible** — plugins, skills, agent profiles, subagents, and MCP clients/servers
- **CLI included** — interactive chat and one-shot coding tasks

## Requirements

- Python 3.10+
- An API key for a supported LiteLLM provider (optional for offline/mock usage)

## Installation

Install the SDK and development tools:

```bash
python -m pip install -e ".[dev]"
```

To use LiteLLM providers:

```bash
python -m pip install -e ".[dev,llm]"
```

Optional extras:

```bash
pip install -e ".[tokens]"    # Real tokenizer support
pip install -e ".[browser]"   # Browser tools
pip install -e ".[otel]"      # OpenTelemetry export
pip install -e ".[memory]"    # LiteLLM memory dependencies
pip install -e ".[dev,llm]"   # Development and LLM dependencies
```

## Quick Start

```python
from kinetic_sdk.agent import Agent
from kinetic_sdk.llm.client import LiteLLMClient
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.tool.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echo a message"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, message: str) -> ToolResult:
        return ToolResult(output=message)


llm = LiteLLMClient(model="gpt-4o", api_key="YOUR_API_KEY")
agent = Agent(
    llm=llm,
    tools=[EchoTool()],
    permission_policy=AllowListPolicy(always_allow=["echo"]),
)

print(agent.run("Echo back: hello"))
```

For a runnable offline example:

```bash
python examples/01_quickstart.py
```

More examples are available in [`examples/`](examples).

## CLI

After installation, the `kinetic` command is available:

```bash
kinetic init
kinetic run "Explain this repository"
kinetic chat
kinetic verify
kinetic version
kinetic --help
```

`gow` is an alias for `kinetic`: run `gow` with no arguments to start the
interactive TUI chat (opencode-style), or `gow run "..."` for a one-shot task.

Configure a model or API key:

```bash
kinetic config --set model=gpt-4o
kinetic config --set api_key=YOUR_API_KEY
kinetic config --set base_url=https://api.example.com
kinetic config --get model
```

Or provide values per command:

```bash
kinetic run "Fix the failing tests" --model gpt-4o --api-key YOUR_API_KEY --safe --stream
```

Useful CLI options:

- `--safe` — use a confirmation-oriented deny-by-default policy
- `--yes` — automatically confirm permission prompts
- `--stream` — stream text output
- `--verify` — require verification evidence
- `--base-url` — use an OpenAI-compatible API endpoint

Project settings are stored in `.kinetic/project.json`. Global CLI settings are stored in `~/.kinetic/config.json`.

## Security

- Tools are denied by default unless explicitly allowed
- Use `PermissivePolicy` only in controlled development environments
- Sensitive values are redacted from tool inputs, outputs, logs, and observations
- Never commit API keys or credentials
- Prefer environment variables or the local CLI configuration for secrets

Example:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
kinetic run "Review the current changes" --safe
```

## Testing

```bash
python -m pytest -q
python -m ruff check kinetic_sdk
python -m mypy kinetic_sdk
```

Integration tests that call real providers require an API key and are not run by default.

## Documentation

- [`CHANGELOG.md`](CHANGELOG.md) — release changes
- [`AGENTS.md`](AGENTS.md) — repository-specific agent guidance
- [`docs/`](docs) — benchmark and baseline documentation
- [`examples/`](examples) — runnable SDK examples
- [`benchmarks/`](benchmarks) — benchmark tasks and evaluation fixtures

## License

MIT — see [`LICENSE`](LICENSE).
