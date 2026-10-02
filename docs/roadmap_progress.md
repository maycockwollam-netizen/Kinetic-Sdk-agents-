# Roadmap progress

## 2026-10-02 — Benchmark PR quality gate

- Built deterministic cassette smoke runner and dependency-free comparison gate.
  Main files: `kinetic_sdk/bench/gate.py`, `scripts/bench_run.py`,
  `scripts/bench_compare.py`, and `benchmarks/cassettes/`.
- Added mock baseline (`benchmarks/baselines/baseline.mock.json`), gate and
  nightly matrix configuration, PR smoke and scheduled nightly workflows.
- Added regression tests for offline determinism, passing results, security /
  secret / cost / infrastructure regression failures, and tolerated confidence
  interval noise: `tests/test_bench.py`.
- Commands run: `python -m pytest -q tests/test_bench.py`. Open issue: no real
  provider benchmark was run in this environment because no
  `KINETIC_BENCH_API_KEY` was supplied; the current baseline is explicitly
  mock-only and needs a reviewed nightly real-model baseline before it can
  represent production quality.

- Full `python -m pytest -q` was also attempted after installing `.[dev,llm]`.
  It reached 1235 passed, 2 skipped, and 4 deselected, but three existing
  tiktoken-counter tests could not download `cl100k_base.tiktoken` because the
  environment's proxy TLS certificate verification failed. This is an
  environment limitation, not a model benchmark run.
