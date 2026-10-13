#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -u -m scripts.mix_match_test_set \
    --currents "$repo_root/runs/current_examples_sb9eo4qc" \
    --history-batches "$repo_root/runs/history_batches_BbuBNu" \
    --output "$repo_root/datasets/test_set_v1" --count "${COUNT:-5}" "$@"
