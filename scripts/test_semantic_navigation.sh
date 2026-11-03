#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 SOURCE_EPISODE [--target cup|bowl] [--object-map hydra|visual] [--seconds 180] [--explorer tare|frontier]" >&2
  exit 2
fi
source_episode="$1"
shift
mkdir -p runs
output_parent=$(mktemp -d "$PWD/runs/semantic_navigation_XXXXXX")
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.venv-vggt-cache/matplotlib}"
export PYTHONPATH="$PWD/external/DAAAM/src:$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
echo "OUTPUT=$output_parent/episode"
exec .venv-slam/bin/python scripts/demo_semantic_navigation.py --source "$source_episode" --output "$output_parent/episode" "$@"
