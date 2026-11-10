#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SENSOR_MODEL_ENDPOINTS:?Set SENSOR_MODEL_ENDPOINTS to the existing model-services JSON}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$PWD:$PWD/external/cap-x:$PWD/external/DAAAM/src:$PWD/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
exec "${QWEN_CAPX_PYTHON:-$PWD/.venv-sift/bin/python}" -u scripts/test_qwen_capx_pick.py "$@"
