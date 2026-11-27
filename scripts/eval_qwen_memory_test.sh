#!/usr/bin/env bash
# Test-set eval of the Qwen + memory planner (the final35 setup), one run per GPU.
#
#   bash scripts/eval_qwen_memory_test.sh                       # all five tasks, test_same_map + test_new_map (300)
#   SPLIT=test_same_map bash scripts/eval_qwen_memory_test.sh   # same layout only (200)
# New-layout houses (no history of the house): text memory from OTHER houses (task procedure + object starts), live explore,
# then cross-room replay of the explore path and fixture-relative open/close/place replays (baseline fallback).
#
# Each episode: --planner qwen_replan, Gemini procedure text memory of its own house and task
# (--text-memory auto, TEXT_MEMORY_STYLE=procedure), memory navigation + skip explore when the
# house is recognized, limits 45 wall min / 20 simulated min / 100 calls. Score = private
# report.json success; summary.json and summary_by_rooms.json in $OUTPUT.
# Env: DATA (the --root of prepare_eval_data.py, with --memory; default runs/all_open_fixed_v4), GPUS (default 0),
# SPLIT (test), TASKS (all), OUTPUT (default runs/household_eval/qwen_memory_<split>_<time>),
# INTO (rerun into an existing batch folder: finished episodes are skipped).
# Extra arguments go to eval_households.py (e.g. --list, --raw, episode folders).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPUS=${GPUS:-0};SPLIT=${SPLIT:-test};TASKS=${TASKS:-all};DATA=${DATA:-$ROOT/runs/all_open_fixed_v4}
args=(--memory --planner qwen_replan --split "$SPLIT" --tasks "$TASKS" --gpus "$GPUS" --dataset "$DATA")
if [ -n "${INTO:-}" ]; then args+=(--into "$INTO")
else args+=(--output "${OUTPUT:-$ROOT/runs/household_eval/qwen_memory_${SPLIT}_$(date +%Y%m%d_%H%M%S)}")
fi
exec "${PYTHON:-$ROOT/.venv-sift/bin/python}" "$ROOT/scripts/eval_households.py" "${args[@]}" "$@"
