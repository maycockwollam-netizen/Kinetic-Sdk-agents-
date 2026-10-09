#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:=mock}"
: "${PROFILE:=mock}"
: "${RUNS:=1}"
export MODEL PROFILE RUNS
exec "${PYTHON:-python}" scripts/run_baseline.py
