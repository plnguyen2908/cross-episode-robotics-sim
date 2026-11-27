#!/usr/bin/env bash
# GPT (gpt-6.1-sol) + new-layout memory on two DEV new-layout episodes, the same ones the Qwen + memory dev runs used
# (house 3d5fe88b, 3 rooms, robot starts in the kitchen): prepare_and_serve_meal_000001 and set_table_000004.
#
#   export OPENAI_API_KEY=...
#   DATA=<dataset with dev_new_map> bash scripts/check_gpt_newlayout_two.sh   # GPUS=0 by default
# (prepare_eval_data.py --splits dev_new_map downloads that split)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
DATA=${DATA:-$ROOT/runs/all_open_fixed_v4};D=$DATA/dev_new_map/batches/round_continuous
exec "${PYTHON:-$ROOT/.venv-sift/bin/python}" "$ROOT/scripts/eval_households.py" --memory --planner gpt_replan --gpus "${GPUS:-0}" \
  --dataset "$DATA" \
  --output "${OUTPUT:-$ROOT/runs/household_eval/gpt_newlayout_check2_$(date +%Y%m%d_%H%M%S)}" \
  $D/prepare_and_serve_meal_000001 $D/set_table_000004 "$@"
