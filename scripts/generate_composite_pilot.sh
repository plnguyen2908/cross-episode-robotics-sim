#!/usr/bin/env bash
# Independent episodes; no history-batch dependency. Requires the shared physical scene adapter.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ROBOCASA_DIR="${ROBOCASA_DIR:-$repo_dir/external/robocasa/robocasa}"
exec "$python_bin" -u -m cross_episode_sim.generation.composite_pilot "$@"
