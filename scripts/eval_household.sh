#!/usr/bin/env bash
# Run one saved household episode with our Qwen sensor pipeline from THIS checkout,
# in the foreground, with a live log in the terminal. No wall-time or sim-time limit.
#
#   bash scripts/eval_household.sh <episode folder> [--gpu N] [--output DIR] [--raw]
#
# Tasks: set_table, warm_two_pots, unpack_and_sort_groceries (and the other household
# tasks). The terminal shows stages, public calls and the private task progress after
# every call (TASK_PROGRESS_AFTER_CALL); --raw shows everything. The full log is always
# <output>/live.log; the score is <output>/episode/report.json (success, checks, progress).
# Needs the model services (Qwen 51500, Molmo 51501, FastSAM 51502, M2T2 51503,
# GraspGen-X 51586) and .venv-sift / external/* set up as for the fix trees.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source_dir="";gpu=2;output="";raw=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu=$2;shift 2;;
    --output) output=$2;shift 2;;
    --raw) raw=1;shift;;
    -h|--help) sed -n 2,12p "$0";exit 0;;
    *) source_dir=$1;shift;;
  esac
done
[ -n "$source_dir" ] && [ -f "$source_dir/config.json" ] || { echo "usage: $0 <episode folder with config.json> [--gpu N] [--output DIR] [--raw]";exit 2; }
source_dir=$(cd "$source_dir" && pwd)
py="$ROOT/.venv-sift/bin/python"
[ -x "$py" ] || { echo "Missing $py (link it from a fix tree)";exit 2; }
task=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['task'])" "$source_dir/config.json")
case "$task" in
  set_table) runner=scripts/run_target_conditioned_household.py;;
  *) runner=scripts/run_sensor_household.py;;
esac
# Same model endpoints as the passing runs (fix186 pots, fix187 set table, fix190 grocery).
endpoints='{"qwen":"127.0.0.1:51500","molmo":"127.0.0.1:51501","fastsam":"127.0.0.1:51502","m2t2":"127.0.0.1:51503","graspgenx":"127.0.0.1:51586"'
[ "$task" = warm_two_pots ] || endpoints="$endpoints"',"onering":"scan"'
endpoints="$endpoints}"
[ -n "$output" ] || output="$ROOT/runs/household_eval/$(basename "$source_dir")_$(date +%Y%m%d_%H%M%S)"
[ -e "$output" ] && { echo "Output exists: $output";exit 2; }
mkdir -p "$output"
commit=$(git -C "$ROOT" rev-parse --short HEAD)
dirty=$(git -C "$ROOT" status --porcelain -- cross_episode_sim scripts | grep -v "^??" | grep -c "\.py$" || true)
python3 - "$output/run_info.json" "$commit" "$dirty" "$source_dir" "$task" "$gpu" "$runner" "$endpoints" <<'EOF'
import json,sys
out,commit,dirty,source,task,gpu,runner,endpoints=sys.argv[1:]
json.dump(dict(commit=commit,modified_code_files=int(dirty),source=source,task=task,gpu=gpu,runner=runner,
               endpoints=json.loads(endpoints),limits='none'),open(out,'w'),indent=2)
EOF
echo "== $task  $(basename "$source_dir")  main@$commit$([ "$dirty" -gt 0 ] && echo " (+$dirty modified code files)")  GPU $gpu"
echo "== log: $output/live.log"
cd "$ROOT"
set +e
env -u HOUSEHOLD_PLAN_COUNTS -u HOUSEHOLD_PLAN_CALLS CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TZ=America/New_York \
  PYTHONPATH="$ROOT:$ROOT/external/DAAAM/src:$ROOT/external/molmospaces" SENSOR_MODEL_ENDPOINTS="$endpoints" \
  "$py" -u "$runner" --source "$source_dir" --output "$output/episode" --skip-video-render 2>&1 \
  | tee "$output/live.log" \
  | if [ "$raw" = 1 ]; then cat; else grep --line-buffered -aE 'STAGE [0-9]+|PUBLIC CALL|TASK_PROGRESS|Traceback|Error|FAILED'; fi
code=${PIPESTATUS[0]}
set -e
python3 - "$output/episode/report.json" <<'EOF'
import json,sys
try:r=json.load(open(sys.argv[1]))
except Exception as exc:print('== no report:',exc);sys.exit()
p=r.get('progress') or {}
print(f"== success {r.get('success')}  progress {p.get('done')}/{p.get('total')} (rate {p.get('rate')})  "
      f"calls {r.get('completed_calls')}/{r.get('planned_calls')}  wall {round((r.get('wall_seconds') or 0)/60,1)} min")
print('== checks',r.get('checks'))
for f in r.get('failed_calls') or []:print('== failed call',f)
EOF
exit $code
