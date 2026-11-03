#!/usr/bin/env bash
# Fresh-map approach -> perception -> docking -> placement, with checked gates.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" MUJOCO_GL=egl OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/nobackup2/le/.cache}"
export PYTHONPATH="$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
start=${1:?held checkpoint}; payload=${2:?observed payload tool points}; out=${3:?new output directory}
mkdir "$out"
.venv-slam/bin/python scripts/capture_loaded_scene.py --start "$start" --payload "$payload" --output "$out/initial_perception"
payload="$out/initial_perception/observed_payload_tool.npy"
.venv-slam/bin/python scripts/test_m2t2_ring_approach.py --start "$start" --payload "$payload" --output "$out/approach" --checkpoint /nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt
approach="$out/approach"
status=$(.venv-slam/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$approach/report.json")
.venv-slam/bin/python scripts/test_m2t2_place.py --dock "$approach" --observe-only --output "$out/perception"
observed=$(.venv-slam/bin/python -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("observation_success",False)))' "$out/perception/report.json")
perception="$out/perception"
if [[ "$status" == "needs_floor_observation" ]]; then
 .venv-slam/bin/python scripts/capture_loaded_scene.py --start "$approach" --payload "$approach/observed_payload_tool.npy" --output "$out/scan_input"
 .venv-slam/bin/python scripts/scan_m2t2_placement_space.py --pick "$approach" --perception "$out/scan_input" --output "$out/floor_scan"
 .venv-slam/bin/python -c 'import json,sys; assert json.load(open(sys.argv[1]))["success"], "Floor scan failed; not continuing"' "$out/floor_scan/report.json"
 .venv-slam/bin/python scripts/test_m2t2_ring_approach.py --start "$out/floor_scan" --payload "$out/floor_scan/observed_payload_tool.npy" --observed-map "$out/floor_scan" --output "$out/approach_retry" --checkpoint /nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt
 approach="$out/approach_retry"
 perception="$out/perception_retry"
 .venv-slam/bin/python scripts/test_m2t2_place.py --dock "$approach" --observe-only --output "$perception"
fi
.venv-slam/bin/python -c 'import json,sys; assert json.load(open(sys.argv[1])).get("observation_success"), "No verified table surface"' "$perception/report.json"
# Perception is in the final base frame; re-anchor the observed map separately.
.venv-slam/bin/python scripts/test_m2t2_place_redock_astar.py --start "$approach" --perception "$perception" --continuation-map "$approach" --output "$out/dock"
.venv-slam/bin/python scripts/test_m2t2_place.py --dock "$out/dock" --output "$out/place"
