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
# Env: GPUS (default 2,5,6,7), SPLIT (test), TASKS (all), OUTPUT (new folder on /nobackup3),
# INTO (rerun into an existing batch folder: finished episodes are skipped).
# Extra arguments go to eval_households.py (e.g. --list, --raw, episode folders).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPUS=${GPUS:-2,5,6,7};SPLIT=${SPLIT:-test};TASKS=${TASKS:-all}
args=(--qwen-memory --split "$SPLIT" --tasks "$TASKS" --gpus "$GPUS" --dataset /nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4)
if [ -n "${INTO:-}" ]; then args+=(--into "$INTO")
else args+=(--output "${OUTPUT:-/nobackup3/le/household_eval/qwen_memory_${SPLIT}_$(date +%Y%m%d_%H%M%S)}")
fi
exec python3 "$ROOT/scripts/eval_households.py" "${args[@]}" "$@"
