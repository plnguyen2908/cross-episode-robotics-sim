#!/usr/bin/env bash
# Sample matching histories and generate complete current examples.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/sample_current_tasks.sh --history-batches PATH [PATH ...] [options]" >&2
    echo "Defaults: sample and generate 30 breakfast_with_coffee examples with 1–3 people inferred from sampled histories." >&2
    exit 2
fi
exec "$PYTHON_BIN" -u -m scripts.sample_current_tasks \
    --config "$repo_root/examples/generation/sample_breakfast_with_coffee.json" \
    --count "${COUNT:-30}" --execute \
    "$@"
