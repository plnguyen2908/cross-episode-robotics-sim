#!/usr/bin/env bash
# One household episode with GPT as the replanning planner (--planner gpt_replan),
# using model services that are already running (default addresses, or
# SERVICE_ENDPOINTS_FILE from scripts/start_eval_services.sh). Live log in the
# terminal; at the end it prints the score, the plan and every replan, and the
# API tokens and cost.
#
#   export OPENAI_API_KEY=...            # read from the environment only, never written
#   bash scripts/run_gpt_replan_example.sh [episode folder] [--gpu N] [--out DIR] [extra eval_household.sh args]
#
# Defaults: the 2-room grocery test episode unpack_and_sort_groceries_000032
# (qwen_replan scored 4/4 on it with 1 replan; open-loop Qwen 3/4), GPU 3,
# the same limits as the batch (--gpt-limits: 45 wall min, 15 simulated min, 100 calls).
# PLANNER_MODEL (default gpt-6.1-sol) and PLANNER_EFFORT (default high) choose the model.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
episode=/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/test_same_map/batches/round_continuous/unpack_and_sort_groceries_000032
gpu=3;out="";extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu=$2;shift 2;;
    --out) out=$2;shift 2;;
    -h|--help) sed -n 2,16p "$0";exit 0;;
    --*) extra+=("$1");shift;;
    *) if [ -d "$1" ]; then episode=$1; else extra+=("$1"); fi;shift;;
  esac
done
[ -n "${OPENAI_API_KEY:-}" ] || { echo "export OPENAI_API_KEY first (it is only read from the environment)";exit 2; }
cd "$ROOT"
PYTHONPATH="$ROOT" .venv-sift/bin/python - <<'PY'
import os
from cross_episode_sim.evaluation.gpt_planner import check_planner_model, PlannerServiceError
try:print('planner model:', check_planner_model() or os.environ.get('PLANNER_MODEL', 'gpt-6.1-sol') + ' (key cannot list models; not checked)')
except PlannerServiceError as exc:raise SystemExit(f'planner check failed: {exc}')
PY
batch=${out:-$ROOT/runs/household_eval/gpt_replan_example_$(date +%Y%m%d_%H%M%S)}
name=$(basename "$episode")
mkdir -p "$batch"
limits=(--gpt-limits)
for x in "${extra[@]:-}"; do case "$x" in --wall-minutes|--sim-minutes|--max-calls|--gpt-limits) limits=();; esac; done
set +e
bash scripts/eval_household.sh "$episode" --gpu "$gpu" --planner gpt_replan "${limits[@]}" "${extra[@]}" --output "$batch/$name"
code=$?
set -e
.venv-sift/bin/python scripts/collect_results.py "$batch" >/dev/null || true
.venv-sift/bin/python - "$batch/$name" <<'PY'
import json,sys
from pathlib import Path
episode=Path(sys.argv[1])/'episode'
try:plan=json.loads((episode/'qwen_plan.json').read_text())
except OSError:raise SystemExit('no plan was written (see live.log)')
print('\n== first plan (%d calls, %d requests, model %s)'%(len(plan.get('calls') or []),plan.get('requests',0),plan.get('model')))
for i,c in enumerate(plan.get('calls') or [],1):print(f"  {i:2d} {c.get('skill')} {json.dumps(c.get('arguments'))}")
for r in plan.get('replans') or []:
    print(f"\n== replan after call {r.get('after_call')} ({r.get('requests')} requests): "+str((r.get('history') or [{}])[-1].get('reason'))[:160])
    for i,c in enumerate(r.get('calls') or [],r.get('after_call',0)+1):print(f"  {i:2d} {c.get('skill')} {json.dumps(c.get('arguments'))}")
PY
.venv-sift/bin/python scripts/planner_usage.py "$batch" | python3 -c "import json,sys;u=json.load(sys.stdin);print(f\"\n== API: {u['requests']} requests, {u['input_tokens']} input ({u['cached_input_tokens']} cached), {u['output_tokens']} output ({u['reasoning_tokens']} reasoning) tokens, cost \${u['cost_usd']}\")"
echo "== everything for this episode: $batch/all_results.jsonl"
exit $code
