#!/usr/bin/env bash
# Run one saved household episode with our Qwen sensor pipeline from THIS checkout,
# in the foreground, with a live log in the terminal. No wall-time or sim-time limit.
#
#   bash scripts/eval_household.sh <episode folder> [--gpu N] [--output DIR] [--raw] [--planner recipe|qwen_zero_shot|qwen_zero_shot_pure|qwen_replan|gpt_zero_shot|gpt_replan]
#        [--gpt-limits | --wall-minutes M --sim-minutes M --max-calls N]
#
# Tasks: set_table, warm_two_pots, unpack_and_sort_groceries (and the other household
# tasks). The terminal shows stages, public calls and the private task progress after
# every call (TASK_PROGRESS_AFTER_CALL); --raw shows everything. The full log is always
# <output>/live.log; the score is <output>/episode/report.json (success, checks, progress).
# --planner qwen_zero_shot: Qwen writes the whole plan from the instruction, the skill catalog
# and the planning-contract rules, with up to 2 repair requests (episode/qwen_plan.json); it is
# then executed open loop. --planner qwen_zero_shot_pure: one request, no rules or repairs.
# --planner qwen_replan: as qwen_zero_shot; after a failed call Qwen rewrites the rest of the plan.
# Default is the fixed recipe. Qwen planning goes to its own worker, --planner-endpoint
# (default 127.0.0.1:51510), so plans never queue in front of perception requests.
# Limits (default none): a run that reaches one stops where it is and the live state is scored.
# --gpt-limits = the GPT agent-as-policy eval's budgets: 45 wall min, 15 simulated min, 100 calls.
# Needs the model services (Qwen 51500, Molmo 51501, FastSAM 51502, M2T2 51503,
# GraspGen-X 51586) and .venv-sift / external/* set up as for the fix trees.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source_dir="";gpu=2;output="";raw=0;planner=recipe;planner_endpoint=127.0.0.1:51510;wall_min="";sim_min="";max_calls=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) gpu=$2;shift 2;;
    --output) output=$2;shift 2;;
    --raw) raw=1;shift;;
    --planner) planner=$2;shift 2;;
    --planner-endpoint) planner_endpoint=$2;shift 2;;
    --wall-minutes) wall_min=$2;shift 2;;
    --sim-minutes) sim_min=$2;shift 2;;
    --max-calls) max_calls=$2;shift 2;;
    --gpt-limits) wall_min=45;sim_min=15;max_calls=100;shift;;
    -h|--help) sed -n 2,12p "$0";exit 0;;
    *) source_dir=$1;shift;;
  esac
done
case "$planner" in recipe|qwen_zero_shot|qwen_zero_shot_pure|qwen_replan|gpt_zero_shot|gpt_replan) ;; *) echo "--planner must be recipe, qwen_zero_shot, qwen_zero_shot_pure, qwen_replan, gpt_zero_shot or gpt_replan";exit 2;; esac
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
# Model service addresses: SERVICE_ENDPOINTS_FILE (written by scripts/start_eval_services.sh)
# or the defaults of the shared services on our machine. Qwen planners also get their
# own planner worker (--planner-endpoint / the file's qwen_planner); GPT planners use the API.
endpoints=$(python3 - "$task" "$planner" "$planner_endpoint" "${SERVICE_ENDPOINTS_FILE:-}" <<'PYEOF'
import json,sys
task,planner,planner_endpoint,path=sys.argv[1:]
e={"qwen":"127.0.0.1:51500","molmo":"127.0.0.1:51501","fastsam":"127.0.0.1:51502","m2t2":"127.0.0.1:51503","graspgenx":"127.0.0.1:51586"}
if path:e.update(json.load(open(path)))
planner_address=e.pop("qwen_planner",None) or planner_endpoint
if task!="warm_two_pots":e["onering"]="scan"
if planner.startswith("qwen_"):e["qwen_planner"]=planner_address
print(json.dumps(e))
PYEOF
)
[ -n "$output" ] || output="$ROOT/runs/household_eval/$(basename "$source_dir")_$(date +%Y%m%d_%H%M%S)"
[ -e "$output" ] && { echo "Output exists: $output";exit 2; }
mkdir -p "$output"
commit=$(git -C "$ROOT" rev-parse --short HEAD)
dirty=$(git -C "$ROOT" status --porcelain -- cross_episode_sim scripts | grep -v "^??" | grep -c "\.py$" || true)
python3 - "$output/run_info.json" "$commit" "$dirty" "$source_dir" "$task" "$gpu" "$runner" "$endpoints" "$planner" "$wall_min" "$sim_min" "$max_calls" <<'EOF'
import json,sys
out,commit,dirty,source,task,gpu,runner,endpoints,planner,wall_min,sim_min,max_calls=sys.argv[1:]
json.dump(dict(commit=commit,modified_code_files=int(dirty),source=source,task=task,gpu=gpu,runner=runner,
               endpoints=json.loads(endpoints),planner=planner,
               limits=dict(wall_minutes=float(wall_min) if wall_min else None,sim_minutes=float(sim_min) if sim_min else None,
                           max_calls=int(max_calls) if max_calls else None)),open(out,'w'),indent=2)
EOF
echo "== $task  $(basename "$source_dir")  planner $planner  main@$commit$([ "$dirty" -gt 0 ] && echo " (+$dirty modified code files)")  GPU $gpu"
echo "== limits: wall ${wall_min:-none} min, simulated ${sim_min:-none} min, calls ${max_calls:-none}"
echo "== log: $output/live.log"
limit_args=()
[ -n "$wall_min" ] && limit_args+=(--wall-seconds-limit "$(python3 -c "print(float('$wall_min')*60)")")
[ -n "$sim_min" ] && limit_args+=(--sim-seconds-limit "$(python3 -c "print(float('$sim_min')*60)")")
[ -n "$max_calls" ] && limit_args+=(--max-calls "$max_calls")
cd "$ROOT"
set +e
env -u HOUSEHOLD_PLAN_COUNTS -u HOUSEHOLD_PLAN_CALLS CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TZ=America/New_York \
  PYTHONPATH="$ROOT:$ROOT/external/DAAAM/src:$ROOT/external/molmospaces" SENSOR_MODEL_ENDPOINTS="$endpoints" \
  "$py" -u "$runner" --source "$source_dir" --output "$output/episode" --skip-video-render --planner "$planner" "${limit_args[@]}" 2>&1 \
  | tee "$output/live.log" \
  | if [ "$raw" = 1 ]; then cat; else grep --line-buffered -aE 'STAGE [0-9]+|PUBLIC CALL|QWEN|TASK_PROGRESS|EPISODE_LIMIT|Traceback|Error|FAILED'; fi
code=${PIPESTATUS[0]}
set -e
python3 - "$output/episode/report.json" <<'EOF'
import json,sys
try:r=json.load(open(sys.argv[1]))
except Exception as exc:print('== no report:',exc);sys.exit()
p=r.get('progress') or {}
print(f"== termination {r.get('termination')}  simulated {round((r.get('simulated_seconds') or 0)/60,1)} min")
print(f"== success {r.get('success')}  progress {p.get('done')}/{p.get('total')} (rate {p.get('rate')})  "
      f"calls {r.get('completed_calls')}/{r.get('planned_calls')}  wall {round((r.get('wall_seconds') or 0)/60,1)} min")
print('== checks',r.get('checks'))
for f in r.get('failed_calls') or []:print('== failed call',f)
EOF
exit $code
