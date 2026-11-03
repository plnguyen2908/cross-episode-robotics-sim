#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ]; then echo "Usage: $0 SENSOR_NAVIGATION_EPISODE_WITH_PRIVATE_CHECKPOINT" >&2; exit 2; fi
mkdir -p runs
output_parent=$(mktemp -d "$PWD/runs/observed_arm_XXXXXX")
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
echo "OUTPUT=$output_parent/episode"
exec .venv-slam/bin/python scripts/test_observed_curobo.py --source "$1" --output "$output_parent/episode"
