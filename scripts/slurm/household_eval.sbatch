#!/usr/bin/env bash
#SBATCH --job-name=household-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                 # 80 GB GPUs: 1 for the model services, 1 shared by the episodes
                                     # (SERVICE_SETS=N: N GPUs, one service set + its episodes per GPU)
#SBATCH --cpus-per-task=96           # ~10 cores per running episode + services
#SBATCH --mem=128G                   # ~6 GB per running episode + ~10 GB services
#SBATCH --time=2-00:00:00            # 300 episodes at 8 in parallel: ~30 h; the batch resumes if cut
#SBATCH --output=household-eval-%j.out
#
# Household evaluation on one node. Edit the variables below, then:
#   sbatch scripts/slurm/household_eval.sbatch
# Prerequisites (once, on a login or build node): scripts/setup_eval_env.sh and
# scripts/prepare_eval_data.py --root $DATA (see README "Household evaluation on Slurm").
# For --planner gpt_*, export OPENAI_API_KEY before sbatch (it is inherited, never written).
set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}                # this repository (with .venv-* from setup_eval_env.sh)
DATA=${DATA:?set DATA to the --root used with prepare_eval_data.py}
OUT=${OUT:-$REPO/runs/household_eval/slurm_${SLURM_JOB_ID}}
PLANNER=${PLANNER:-gpt_replan}                 # gpt_replan | gpt_zero_shot | qwen_replan | qwen_zero_shot | recipe
SPLITS=${SPLITS:-test}                         # test = test_same_map,test_new_map (300 episodes)
TASKS=${TASKS:-set_table,clear_and_store,prepare_and_serve_meal,unpack_and_sort_groceries,warm_two_pots}
EPISODES_PER_GPU=${EPISODES_PER_GPU:-8}        # SERVICE_SETS=1: parallel episodes on each episode GPU (CPU-bound)
SERVICE_SETS=${SERVICE_SETS:-1}                # >1: one model-service set per GPU (~41 GB each), each episode
EPISODES_PER_SET=${EPISODES_PER_SET:-1}        #     uses only its own set; EPISODES_PER_SET episodes per set
LIMITS=${LIMITS:---gpt-limits}                 # 45 wall min, 15 simulated min, 100 calls per episode
BASE_PORT=${BASE_PORT:-$((52000 + SLURM_JOB_ID % 1000 * 10))}

cd "$REPO"
# shellcheck disable=SC1091
source eval_env.sh                              # written by scripts/setup_eval_env.sh
# GPUs of this job as CUDA sees them inside the job (Slurm sets CUDA_VISIBLE_DEVICES).
IFS=, read -ra JOB_GPUS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
[ ${#JOB_GPUS[@]} -ge 2 ] || { echo "need 2 GPUs (services + episodes)";exit 2; }
if [ "$SERVICE_SETS" -gt 1 ]; then
  [ ${#JOB_GPUS[@]} -ge "$SERVICE_SETS" ] || { echo "SERVICE_SETS=$SERVICE_SETS needs that many GPUs";exit 2; }
  SET_GPUS=("${JOB_GPUS[@]:0:$SERVICE_SETS}");EPISODE_GPUS=("${SET_GPUS[@]}")
  SERVICE_GPU=$(IFS=,;echo "${SET_GPUS[*]}")
else
  SERVICE_GPU=${JOB_GPUS[0]};EPISODE_GPUS=("${JOB_GPUS[@]:1}")
fi
echo "job $SLURM_JOB_ID on $(hostname): $SERVICE_SETS service set(s) on GPU(s) $SERVICE_GPU, episode GPUs ${EPISODE_GPUS[*]}, planner $PLANNER"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# MuJoCo renders with EGL on the same GPU number; check every episode GPU before
# starting (if this fails on your cluster, see README: EGL under Slurm).
for g in "${EPISODE_GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g .venv-sift/bin/python -c "
import mujoco
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>')
r=mujoco.Renderer(m,32,32);r.update_scene(mujoco.MjData(m));r.render();print('EGL render ok on GPU $g')"
done

mkdir -p "$OUT"
QWEN_PLANNER_FLAG=""; case "$PLANNER" in qwen_*) QWEN_PLANNER_FLAG=--qwen-planner;; esac
bash scripts/start_eval_services.sh --sets "$SERVICE_SETS" --gpus "$SERVICE_GPU" --out "$OUT/services" --base-port "$BASE_PORT" $QWEN_PLANNER_FLAG \
  > "$OUT/services.log" 2>&1 &
SERVICES=$!
trap 'kill $SERVICES 2>/dev/null || true; wait $SERVICES 2>/dev/null || true' EXIT
until grep -q "all services ready" "$OUT/services.log" 2>/dev/null; do
  kill -0 $SERVICES 2>/dev/null || { echo "services failed:";tail -30 "$OUT/services.log";exit 1; }
  sleep 10
done
ENDPOINT_FILES=$(cat "$OUT/services/endpoints_files.txt")
cat "$OUT/services/endpoints.json" 2>/dev/null || true

# One worker per entry. One set: each episode GPU appears EPISODES_PER_GPU times, all
# using the set. Several sets: set k's GPU appears EPISODES_PER_SET times, each with set k.
if [ "$SERVICE_SETS" -gt 1 ]; then
  IFS=, read -ra FILES <<< "$ENDPOINT_FILES";WORKERS="";ENDPOINTS=""
  for k in "${!EPISODE_GPUS[@]}"; do for _ in $(seq "$EPISODES_PER_SET"); do
    WORKERS+="${EPISODE_GPUS[$k]},";ENDPOINTS+="${FILES[$k]},";done;done
  ENDPOINT_ARGS=(--endpoints "${ENDPOINTS%,}")
else
  WORKERS=$(for g in "${EPISODE_GPUS[@]}"; do for _ in $(seq "$EPISODES_PER_GPU"); do printf '%s,' "$g"; done; done)
  ENDPOINT_ARGS=(--endpoints "$ENDPOINT_FILES")
fi
WORKERS=${WORKERS%,}
BATCH=$OUT/batch
RESUME=(); [ -d "$BATCH" ] && RESUME=(--into "$BATCH")   # a re-submitted job continues the same batch
[ -d "$BATCH" ] || RESUME=(--output "$BATCH")
.venv-sift/bin/python scripts/eval_households.py --dataset "$DATA" --split "$SPLITS" --tasks "$TASKS" \
  --gpus "$WORKERS" "${ENDPOINT_ARGS[@]}" --planner "$PLANNER" $LIMITS "${RESUME[@]}"
.venv-sift/bin/python scripts/eval_households.py --rescore "$BATCH"
[ "${PLANNER#gpt_}" != "$PLANNER" ] && .venv-sift/bin/python scripts/planner_usage.py "$BATCH" || true
echo "done: $BATCH (summary.json, summary_by_rooms.json, <episode>/episode/calls.jsonl)"
