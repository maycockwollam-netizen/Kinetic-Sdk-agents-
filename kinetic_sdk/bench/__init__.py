"""Deterministic benchmark primitives used by CI smoke and nightly runs."""

from kinetic_sdk.bench.gate import compare_results, render_markdown

__all__ = ["compare_results", "render_markdown"]
