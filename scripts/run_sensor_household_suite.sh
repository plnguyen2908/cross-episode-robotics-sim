#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/external/DAAAM/src:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
exec "${SENSOR_PYTHON:-.venv-sift/bin/python}" scripts/run_sensor_household_suite.py "$@"
