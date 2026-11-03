#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_run="${1:-runs/docking_distance_smooth1/close_forward}"
output_run="${2:-$(mktemp -d "$PWD/runs/m2t2_pick_XXXXXX")}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
printf 'OUTPUT=%s\n' "$output_run"
.venv-slam/bin/python scripts/capture_m2t2_input.py --source "$source_run" --output "$output_run"
.venv-m2t2/bin/python scripts/m2t2_worker.py --input "$output_run/input.npz" --output "$output_run/predictions.json" \
 --checkpoint "${M2T2_CHECKPOINT:-/nobackup2/le/.cache/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth}"
.venv-slam/bin/python scripts/test_m2t2_pick.py --source "$source_run" --input "$output_run/input.npz" \
 --predictions "$output_run/predictions.json" --output "$output_run/physics"
.venv-slam/bin/python scripts/render_m2t2_geometry.py --pick "$output_run"
