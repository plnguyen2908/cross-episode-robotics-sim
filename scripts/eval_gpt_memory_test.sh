#!/usr/bin/env bash
# Test-set eval of OUR memory pipeline with GPT (gpt-6.1-sol, OpenAI Responses API) as the planner in place of Qwen.
#
#   export OPENAI_API_KEY=...                                   # read from the environment only, never written
#   SPLIT=test_same_map bash scripts/eval_gpt_memory_test.sh    # same layout (200)
#   bash scripts/eval_gpt_memory_test.sh                        # test_same_map + test_new_map (300)
#
# Identical to scripts/eval_qwen_memory_test.sh except the planner: --planner gpt_replan
# (cross_episode_sim/evaluation/gpt_planner.py, the drop-in client of the Slurm release d1a60c8): the same prompt
# (instruction + skill catalog + planning-contract rules + Gemini procedure text memory of the house/task +
# where_to_search), the same gray placeholder image, the same catalog/contract checks with up to 2 repairs,
# the same replans after failed calls, the same executor, memory navigation + skip explore and limits
# (45 wall min / 20 simulated min / 100 calls; GPT_LIMITS=1 for the agent-as-policy budgets 45/15/100).
# Model: PLANNER_MODEL (default gpt-6.1-sol), PLANNER_EFFORT (default high), PLANNER_TIMEOUT (600 s).
# API tokens per request go into each episode's qwen_plan.json (scripts/planner_usage.py totals them).
# Env: GPUS (default 2,5,6: GPU 7 renderer failures 2026-11-26), SPLIT (test), TASKS (all),
# OUTPUT (new folder on /nobackup3), INTO (rerun into an existing batch folder).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
GPUS=${GPUS:-2,5,6};SPLIT=${SPLIT:-test};TASKS=${TASKS:-all}
args=(--qwen-memory --planner gpt_replan --split "$SPLIT" --tasks "$TASKS" --gpus "$GPUS" --dataset /nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4)
[ "${GPT_LIMITS:-0}" = 1 ] && args+=(--gpt-limits)
if [ -n "${INTO:-}" ]; then args+=(--into "$INTO")
else args+=(--output "${OUTPUT:-/nobackup3/le/household_eval/gpt_memory_${SPLIT}_$(date +%Y%m%d_%H%M%S)}")
fi
exec python3 "$ROOT/scripts/eval_households.py" "${args[@]}" "$@"
