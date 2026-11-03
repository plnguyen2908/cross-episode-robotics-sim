#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/nobackup2/le/.cache/openpi}"
export HF_HOME="${HF_HOME:-/nobackup2/le/.cache}"
export WANDB_ENTITY="${WANDB_ENTITY:-plnguyen2908-university-of-wisconsin-madison}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.90}"
# Release initialization temporaries; BFC retention failed full-replica updates.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:$PWD/external/openpi/src"
exec .venv-openpi/bin/python scripts/train_onboard_pi05.py \
 --checkpoint "$OPENPI_DATA_HOME/openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris" "$@"
