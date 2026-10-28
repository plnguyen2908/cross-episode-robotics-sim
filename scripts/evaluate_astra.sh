#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${PYTHON:-/nobackup/le/cross-episode-native-molmospaces/.venv-native/bin/python}" scripts/evaluate_astra.py "$@"
