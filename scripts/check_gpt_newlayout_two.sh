#!/usr/bin/env bash
# GPT (gpt-6.1-sol) + new-layout memory on two DEV new-layout episodes, the same ones the Qwen + memory dev runs used
# (house 3d5fe88b, 3 rooms, robot starts in the kitchen): prepare_and_serve_meal_000001 and set_table_000004.
#
#   export OPENAI_API_KEY=...
#   bash scripts/check_gpt_newlayout_two.sh          # GPUS=2,5 by default (one each, in parallel)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
D=/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/dev_new_map/batches/round_continuous
exec python3 "$ROOT/scripts/eval_households.py" --qwen-memory --planner gpt_replan --gpus "${GPUS:-2,5}" \
  --dataset /nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4 \
  --output "${OUTPUT:-/nobackup3/le/household_eval/gpt_newlayout_check2_$(date +%Y%m%d_%H%M%S)}" \
  $D/prepare_and_serve_meal_000001 $D/set_table_000004 "$@"
