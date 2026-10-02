#!/usr/bin/env python3
"""Compare benchmark JSON documents and emit a Markdown PR gate report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinetic_sdk.bench.gate import compare_results, load_toml_config, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("benchmarks/gate.toml"))
    args = parser.parse_args()
    comparison = compare_results(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.results.read_text(encoding="utf-8")),
        load_toml_config(args.gate),
    )
    print(render_markdown(comparison), end="")
    return 0 if comparison.passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
