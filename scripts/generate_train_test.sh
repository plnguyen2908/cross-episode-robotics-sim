#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ROBOCASA_DIR="${ROBOCASA_DIR:-$repo_dir/external/robocasa/robocasa}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" -m cross_episode_sim.generation.training_dataset "$@"
