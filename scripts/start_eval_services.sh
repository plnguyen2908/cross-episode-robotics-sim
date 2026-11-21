#!/usr/bin/env bash
# Start the model services the household evaluation needs, wait until each is
# ready, and write their addresses to <out>/endpoints.json. Run it in the
# background of the same job as the evaluation, then:
#   export SERVICE_ENDPOINTS_FILE=<out>/endpoints.json
#
#   bash scripts/start_eval_services.sh --gpus 0 --out runs/services [--base-port 52000] [--qwen-planner]
#
# Services (gRPC, localhost):            GPU memory   weights (downloaded to HF_HOME)
#   qwen      perception VLM              ~23 GB       Qwen/Qwen3-VL-8B-Instruct
#   molmo     pointing                    ~11 GB       allenai/Molmo2-4B
#   fastsam   segmentation                 ~4 GB       external/DAAAM/checkpoints/fastsam/FastSAM-s.pt
#   m2t2      grasp proposals              ~1 GB       wentao-yuan/m2t2 (m2t2.pth)
#   graspgenx grasp proposals              ~2 GB       GraspGen-X checkpoints (setup_eval_env.sh)
#   qwen_planner (only --qwen-planner,    ~20 GB       Qwen/Qwen3-VL-8B-Instruct, a second copy
#     i.e. --planner qwen_*; GPT planners use the OpenAI API instead)
# GPU placement: one GPU (80 GB) holds all of them (~41 GB, ~61 GB with
# --qwen-planner). With several GPUs: qwen on the 1st, molmo/fastsam/m2t2/
# graspgenx on the 2nd, qwen_planner on the 3rd (fewer GPUs wrap around).
# Ctrl-C / SIGTERM stops every service this script started.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Paths and caches written by scripts/setup_eval_env.sh (HF_HOME, checkpoints, ...).
# shellcheck disable=SC1091
[ -f "$ROOT/eval_env.sh" ] && source "$ROOT/eval_env.sh"
gpus=0;out="";base=52000;planner=0;timeout=3600
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) gpus=$2;shift 2;;
    --out) out=$2;shift 2;;
    --base-port) base=$2;shift 2;;
    --qwen-planner) planner=1;shift;;
    --ready-timeout) timeout=$2;shift 2;;
    -h|--help) sed -n 2,24p "$0";exit 0;;
    *) echo "unknown argument $1";exit 2;;
  esac
done
[ -n "$out" ] || { echo "--out is required";exit 2; }
mkdir -p "$out";out=$(cd "$out" && pwd)
IFS=, read -ra G <<< "$gpus"
gpu(){ local i=$1;echo "${G[$(( i % ${#G[@]} ))]}"; }
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export PYTHONPATH="$ROOT:$ROOT/external/DAAAM/src:$ROOT/external/molmospaces"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$ROOT"
FASTSAM=${FASTSAM_CHECKPOINT:-$ROOT/external/DAAAM/checkpoints/fastsam/FastSAM-s.pt}
[ -f "$FASTSAM" ] || { echo "Missing $FASTSAM (run scripts/setup_eval_env.sh)";exit 2; }
M2T2=${M2T2_CHECKPOINT:-$(.venv-sift/bin/python -c "from huggingface_hub import hf_hub_download;print(hf_hub_download('wentao-yuan/m2t2','m2t2.pth'))")}
: > "$out/pids"
cleanup(){ while read -r pid;do kill "$pid" 2>/dev/null || true;done < "$out/pids"; }
trap cleanup INT TERM
start(){ # name kind gpu port python [extra args]
  local name=$1 kind=$2 g=$3 port=$4 py=$5;shift 5
  case " ${SKIP_SERVICES:-} " in *" $name "*) echo "skipping $name (SKIP_SERVICES)";return;; esac
  echo "starting $name on GPU $g port $port"
  CUDA_VISIBLE_DEVICES=$g nohup "$py" -u scripts/serve_sensor_model.py --kind "$kind" --port "$port" --device cuda:0 \
    --output "$out/$name" "$@" > "$out/$name.log" 2>&1 &
  echo $! >> "$out/pids";echo "$name $port $!" >> "$out/services.txt"
}
: > "$out/services.txt"
start qwen      qwen      "$(gpu 0)" $((base+0)) .venv-sift/bin/python
start molmo     molmo     "$(gpu 1)" $((base+1)) .venv-sift/bin/python
start fastsam   fastsam   "$(gpu 1)" $((base+2)) .venv-sift/bin/python --checkpoint "$FASTSAM"
start m2t2      m2t2      "$(gpu 1)" $((base+3)) .venv-m2t2/bin/python --checkpoint "$M2T2"
start graspgenx graspgenx "$(gpu 1)" $((base+4)) .venv-graspgenx/bin/python
[ "$planner" = 1 ] && start qwen_planner qwen "$(gpu 2)" $((base+5)) .venv-sift/bin/python
# Wait for "READY <kind> <address>" from every service (first start downloads weights).
deadline=$(( $(date +%s) + timeout ))
while read -r name port pid; do
  until grep -q "^READY" "$out/$name.log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "service $name exited; last lines of $out/$name.log:";tail -20 "$out/$name.log";cleanup;exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "service $name not ready after ${timeout}s";cleanup;exit 1; }
    sleep 5
  done
  echo "ready: $name 127.0.0.1:$port"
done < "$out/services.txt"
python3 - "$out" <<'PY'
import json,sys
out=sys.argv[1]
endpoints={name:f'127.0.0.1:{port}' for name,port,_ in (l.split() for l in open(out+'/services.txt'))}
json.dump(endpoints,open(out+'/endpoints.json','w'),indent=2);print(json.dumps(endpoints))
PY
echo "all services ready; export SERVICE_ENDPOINTS_FILE=$out/endpoints.json"
# Keep running so the services live as long as this script (stop with Ctrl-C / kill).
wait
