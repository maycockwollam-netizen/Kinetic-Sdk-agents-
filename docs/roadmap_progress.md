# Roadmap progress

## 2026-10-02 — Coding benchmark scaffold

- Built the independent `kinetic_sdk.bench` framework: strict task loading,
  isolated repeated runs, post-run hidden-test injection, independent graders,
  JSON records, and JSON/Markdown reporting.
- Main paths: `kinetic_sdk/bench/task.py`, `kinetic_sdk/bench/grader.py`,
  `kinetic_sdk/bench/runner.py`, `kinetic_sdk/bench/report.py`, and
  `kinetic_sdk/bench/__main__.py`.
- Added the first three fixtures and oracle patches under `benchmarks/tasks/`,
  framework/oracle tests in `tests/test_bench*.py`, and operator documentation
  in `docs/benchmark.md`.
- Ran: `python -m pytest -q tests/test_bench.py tests/test_bench_tasks.py`,
  `ruff check kinetic_sdk tests`, `mypy kinetic_sdk`, and a 3×3 mock CLI run.
- Open issues: the mock CLI currently supplies scripted final text only; a
  tool-calling script format and a first-class Docker factory/configuration
  are future work. Full suite could not be verified in this environment
  because installed pytest lacks `pytest-asyncio` and `litellm`.
