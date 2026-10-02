# Coding benchmarks

`kinetic_sdk.bench` runs every coding task repeatedly in a newly copied fixture,
then grades the resulting workspace **after** the agent exits. Hidden tests are
copied only at this post-run point. The harness uses only the standard library.

## Task schema

Each `benchmarks/tasks/<id>/task.toml` requires `id`, `title`, `category`,
`difficulty`, `prompt`, and `fixture_dir`. Optional keys are
`hidden_tests_dir`, `allowed_tools`, `policy_profile`, `risk_tags`, `budget`,
and `graders`. Unknown fields are errors. `budget` accepts positive
`max_llm_calls`, `max_total_tokens`, `max_tool_calls`, and `max_wall_seconds`.

Graders are either `[graders.tests]`/`[graders.command]` tables or `[[graders]]`
with `type`. Supported kinds are `tests`, `command`, `diff_constraints`,
`no_secret_leak`, and `policy_denied`.

```toml
id = "my_fix"
title = "Fix parser"
category = "bugfix"
difficulty = "easy"
prompt = "Fix the parser and run tests."
fixture_dir = "repo"
hidden_tests_dir = "hidden_tests"
[budget]
max_llm_calls = 8
[[graders]]
type = "tests"
command = "python -m pytest -q"
```

Place a minimal repository in `repo/`, private assertions in `hidden_tests/`,
and an independently reviewable `solution.patch`. Apply the patch in a fresh
fixture and verify the graders pass before adding a task.

## Run and report

```bash
python -m kinetic_sdk.bench run --tasks benchmarks/tasks --runs 3 --mock-script script.json --unsafe-local-exec --out results.json
python -m kinetic_sdk.bench report results.json
```

Production execution must provide a Docker-backed agent/workspace; host command
execution is intentionally an explicit `--unsafe-local-exec` escape hatch for
trusted fixtures and CI mocks. The current mock CLI is text-only; use the
Python `BenchRunner` factory API to construct an agent with Docker tools or a
scripted tool-calling client.
