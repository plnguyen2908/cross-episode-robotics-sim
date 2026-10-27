#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/leos/.venvs/sim/bin/python}
bash scripts/setup_molmospaces_controllers.sh
export PYTHONPATH="$PWD/external/molmospaces:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
if [[ -z "${OUTPUT:-}" ]]; then OUTPUT=$(mktemp -d "$PWD/runs/molmospaces_absolute_XXXXXX"); fi
SOURCE=${SOURCE:-runs/train_test_v1/train/batches/round_0162/clear_and_store_000162}
echo "BACKEND=molmospaces OUTPUT=$OUTPUT/episode"
exec "$PYTHON" scripts/replay_molmospaces_absolute.py --source "$SOURCE" --output "$OUTPUT/episode" "$@"
