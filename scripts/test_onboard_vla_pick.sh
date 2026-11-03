#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-/nobackup2/le/.cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/nobackup2/le/.cache/openpi}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-slam/bin/python scripts/test_onboard_vla_pick.py \
  --checkpoint "${CHECKPOINT:-$OPENPI_DATA_HOME/openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris}" "$@"
