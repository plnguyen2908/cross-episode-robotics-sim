#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -u -m scripts.evaluate_eqa --dataset "$repo_root/datasets/test_set_v1" "$@"
