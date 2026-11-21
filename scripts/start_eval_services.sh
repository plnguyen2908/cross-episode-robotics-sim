#!/usr/bin/env bash
# Start the model services the household evaluation needs, wait until each is
# ready, and write their addresses to <out>/endpoints.json. Run it in the
# background of the same job as the evaluation, then:
#   export SERVICE_ENDPOINTS_FILE=<out>/endpoints.json
#
#   bash scripts/start_eval_services.sh --gpus 0 --out runs/services [--base-port 52000] [--qwen-planner]
#   bash scripts/start_eval_services.sh --sets 8 --gpus 0,1,2,3,4,5,6,7 --out runs/services   # 8 service sets
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
# --sets N starts N independent copies ("service sets"), set k entirely on GPU
# k of --gpus (cycled) with ports base+10k..base+10k+5 and its own
# <out>/set_<k>/endpoints.json, so each running episode can use its own set
# (scripts/eval_households.py --endpoints). One set needs ~41 GB (~61 GB with
# --qwen-planner): one set per 80 GB GPU, plus that GPU's episodes (~1.2 GB each).
# Ctrl-C / SIGTERM stops every service this script started.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Paths and caches written by scripts/setup_eval_env.sh (HF_HOME, checkpoints, ...).
# shellcheck disable=SC1091
[ -f "$ROOT/eval_env.sh" ] && source "$ROOT/eval_env.sh"
gpus=0;out="";base=52000;planner=0;timeout=3600;sets=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) gpus=$2;shift 2;;
    --out) out=$2;shift 2;;
    --base-port) base=$2;shift 2;;
    --qwen-planner) planner=1;shift;;
    --sets) sets=$2;shift 2;;
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
: > "$out/pids";: > "$out/services.txt"
cleanup(){ while read -r pid;do kill "$pid" 2>/dev/null || true;done < "$out/pids"; }
trap cleanup INT TERM
start(){ # set dir name kind gpu port python [extra args]
  local set=$1 dir=$2 name=$3 kind=$4 g=$5 port=$6 py=$7;shift 7
  case " ${SKIP_SERVICES:-} " in *" $name "*) echo "skipping $name (SKIP_SERVICES)";return;; esac
  echo "starting set $set $name on GPU $g port $port"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$g nohup "$py" -u scripts/serve_sensor_model.py --kind "$kind" --port "$port" --device cuda:0 \
    --output "$dir/$name" "$@" > "$dir/$name.log" 2>&1 &
  echo $! >> "$out/pids";echo "$set $name $port $! $dir/$name.log" >> "$out/services.txt"
}
for ((k=0; k<sets; k++)); do
  if [ "$sets" = 1 ]; then dir=$out;g0=$(gpu 0);g1=$(gpu 1);g2=$(gpu 2)   # one set: spread over the GPUs given
  else dir=$out/set_$k;g0=$(gpu $k);g1=$g0;g2=$g0;fi                     # several: set k on one GPU
  p=$((base + 10*k))
  start $k "$dir" qwen      qwen      "$g0" $((p+0)) .venv-sift/bin/python
  start $k "$dir" molmo     molmo     "$g1" $((p+1)) .venv-sift/bin/python
  start $k "$dir" fastsam   fastsam   "$g1" $((p+2)) .venv-sift/bin/python --checkpoint "$FASTSAM"
  start $k "$dir" m2t2      m2t2      "$g1" $((p+3)) .venv-m2t2/bin/python --checkpoint "$M2T2"
  start $k "$dir" graspgenx graspgenx "$g1" $((p+4)) .venv-graspgenx/bin/python
  if [ "$planner" = 1 ]; then start $k "$dir" qwen_planner qwen "$g2" $((p+5)) .venv-sift/bin/python; fi
done
# Wait for "READY <kind> <address>" from every service (first start downloads weights).
deadline=$(( $(date +%s) + timeout ))
while read -r set name port pid log; do
  until grep -q "^READY" "$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "service $name (set $set) exited; last lines of $log:";tail -20 "$log";cleanup;exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "service $name (set $set) not ready after ${timeout}s";cleanup;exit 1; }
    sleep 5
  done
  echo "ready: set $set $name 127.0.0.1:$port"
done < "$out/services.txt"
python3 - "$out" "$sets" <<'PY'
import json,sys,collections
out,sets=sys.argv[1],int(sys.argv[2])
by=collections.defaultdict(dict)
for line in open(out+'/services.txt'):
    k,name,port,_,log=line.split()
    by[int(k)][name]=f'127.0.0.1:{port}'
files=[]
for k in range(sets):
    path=out+'/endpoints.json' if sets==1 else f'{out}/set_{k}/endpoints.json'
    json.dump(by[k],open(path,'w'),indent=2);files.append(path)
open(out+'/endpoints_files.txt','w').write(','.join(files)+'\n')
print(json.dumps(by))
PY
echo "all services ready; endpoints files: $(cat "$out/endpoints_files.txt")"
# Keep running so the services live as long as this script (stop with Ctrl-C / kill).
wait
