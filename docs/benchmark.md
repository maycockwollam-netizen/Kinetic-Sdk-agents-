# Benchmark quality gate

`kinetic_sdk/bench/` supplies a dependency-free quality gate for agent benchmark
results. It deliberately separates deterministic PR smoke coverage from real
model-quality measurement: a pull request must never spend provider credits or
need a secret to prove that the harness, agent loop, policy, and graders work.

## Result format and baseline

A result document is JSON with `schema_version: 1` and one entry per task:
`id`, `groups`, `success`, `cost_usd`, `secret_leaks`, and `infra_error`.
`benchmarks/baselines/baseline.mock.json` is an intentionally mock-only
baseline built from `benchmarks/cassettes/`; it validates the gate plumbing,
not real model quality. A production baseline should be promoted from reviewed
nightly output before using a real-model gate.

Run offline smoke manually:

```bash
python scripts/bench_run.py --mock-script benchmarks/cassettes --output benchmarks/results-smoke.json
python scripts/bench_compare.py benchmarks/baselines/baseline.mock.json benchmarks/results-smoke.json
```

Cassettes contain recorded `MockLLMClient` response turns, including a tool
loop and a security/red-team case. They are deterministic and do not create a
network client.

## Gate rules

Settings live in `benchmarks/gate.toml`. The gate fails when:

- any task in the `security` or `red-team` group fails;
- any reported secret canary leak is non-zero;
- success rate drops more than `success_rate_drop` **and** falls below the
  baseline Wilson 95% confidence interval lower bound;
- total cost divided by successful tasks increases more than the configured
  percentage; or
- `infra_error` count grows beyond its separate allowance.

The confidence check prevents a small sample fluctuation from being blamed on
an agent regression. The report lists changed task IDs and every concrete
failure reason. Infra failures are kept separate because they indicate a
broken benchmark/harness rather than a model-quality verdict.

## CI

`.github/workflows/bench-smoke.yml` runs for every pull request using only the
mock cassettes, writes the Markdown gate report to `GITHUB_STEP_SUMMARY`, and
uploads the JSON result as an artifact. `.github/workflows/bench-nightly.yml`
runs nightly or on dispatch; it reads the model × FLASH/MAX profile matrix from
`benchmarks/matrix.toml`, and runs only when `KINETIC_BENCH_API_KEY` is
configured. Its result plus a timestamped `benchmarks/history.jsonl` line are
uploaded as artifacts; the workflow never commits history automatically.
