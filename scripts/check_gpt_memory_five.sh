#!/usr/bin/env bash
# One episode per task with GPT (gpt-6.1-sol) + our memory: the five same-layout episodes of the Qwen + memory 4/5
# result (final35): set_table_000024, clear_and_store_000024, prepare_and_serve_meal_000027,
# unpack_and_sort_groceries_000032, warm_two_pots_000033. Same setup as scripts/eval_gpt_memory_test.sh.
#
#   export OPENAI_API_KEY=...
#   DATA=<dataset> bash scripts/check_gpt_memory_five.sh   # GPUS=0 by default (one at a time)
#
# Then: python3 scripts/planner_usage.py <output folder>   (API tokens and cost)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
DATA=${DATA:-$ROOT/runs/all_open_fixed_v4};D=$DATA/test_same_map/batches/round_continuous
eps=($D/set_table_000024 $D/clear_and_store_000024 $D/prepare_and_serve_meal_000027 $D/unpack_and_sort_groceries_000032 $D/warm_two_pots_000033)
exec "${PYTHON:-$ROOT/.venv-sift/bin/python}" "$ROOT/scripts/eval_households.py" --memory --planner gpt_replan --gpus "${GPUS:-0}" \
  --dataset "$DATA" \
  --output "${OUTPUT:-$ROOT/runs/household_eval/gpt_memory_check5_$(date +%Y%m%d_%H%M%S)}" "${eps[@]}" "$@"
