#!/usr/bin/env bash
# One episode per task with GPT (gpt-6.1-sol) + our memory: the five same-layout episodes of the Qwen + memory 4/5
# result (final35): set_table_000024, clear_and_store_000024, prepare_and_serve_meal_000027,
# unpack_and_sort_groceries_000032, warm_two_pots_000033. Same setup as scripts/eval_gpt_memory_test.sh.
#
#   export OPENAI_API_KEY=...
#   bash scripts/check_gpt_memory_five.sh            # GPUS=2,5,6 by default (3 at a time, then 2)
#
# Then: python3 scripts/planner_usage.py <output folder>   (API tokens and cost)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
D=/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/test_same_map/batches/round_continuous
eps=($D/set_table_000024 $D/clear_and_store_000024 $D/prepare_and_serve_meal_000027 $D/unpack_and_sort_groceries_000032 $D/warm_two_pots_000033)
exec python3 "$ROOT/scripts/eval_households.py" --qwen-memory --planner gpt_replan --gpus "${GPUS:-2,5,6}" \
  --dataset /nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4 \
  --output "${OUTPUT:-/nobackup3/le/household_eval/gpt_memory_check5_$(date +%Y%m%d_%H%M%S)}" "${eps[@]}" "$@"
