#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_run="${1:-runs/docking_distance_smooth1/close_forward}"
output_run="${2:-$(mktemp -d "$PWD/runs/m2t2_pick_XXXXXX")}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
printf 'OUTPUT=%s\n' "$output_run"
"${SENSOR_PYTHON:-.venv-sift/bin/python}" scripts/capture_m2t2_input.py --source "$source_run" --output "$output_run"
.venv-m2t2/bin/python scripts/m2t2_worker.py --input "$output_run/input.npz" --output "$output_run/predictions.json" \
 --checkpoint "${M2T2_CHECKPOINT:-/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth}"
"${SENSOR_PYTHON:-.venv-sift/bin/python}" scripts/test_m2t2_pick.py --source "$source_run" --input "$output_run/input.npz" \
 --predictions "$output_run/predictions.json" --output "$output_run/physics"
# Video rendering is deferred until the episode ends.
# Standalone: scripts/render_deferred_rollout.py --recording "$output_run/physics/state_recording" --output "$output_run/physics/rollout.mp4"
