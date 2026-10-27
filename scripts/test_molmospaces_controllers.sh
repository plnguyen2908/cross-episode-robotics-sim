#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_molmospaces_controllers.sh
export PYTHONPATH="$PWD/external/molmospaces:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON=${PYTHON:-/home/leos/.venvs/sim/bin/python}
if [[ -z "${OUTPUT:-}" ]]; then
  parent=$(mktemp -d "$PWD/runs/molmospaces_controllers_XXXXXX")
  OUTPUT="$parent/episodes"
fi
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
echo "BACKEND=molmospaces OUTPUT=$OUTPUT"
exec "$PYTHON" scripts/test_molmospaces_controllers.py --output "$OUTPUT" --sources "${SOURCES:-configs/vla/controller_demo_sources.json}" --workers "${WORKERS:-8}" --gpus "$GPUS" "$@"
