#!/usr/bin/env bash
# Qualification runner: sensor navigation then first M2T2 pick, NOT full set-table.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/nobackup2/le/.cache}"
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/external/DAAAM/src:$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces${PYTHONPATH:+:$PYTHONPATH}"
source_episode="${1:-/nobackup/le/cross-episode-robotics-sim/runs/train_test_v1/test_two_rooms/batches/round_continuous/set_table_000002}"
output_run="${2:-$(mktemp -d "$PWD/runs/m2t2_set_table_XXXXXX")}"
mkdir -p "$output_run"
python3 - "$source_episode" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'room_doors.json').read_text());c=json.loads((p/'config.json').read_text())
assert c['task']=='set_table', 'Expected set_table task'
assert d and all(v=='open' for v in d.values()), 'This test requires already-open doors'
print('EPISODE='+str(p));print('PROMPT='+c['instruction'])
PY
printf 'OUTPUT=%s\n' "$output_run"
.venv-slam/bin/python scripts/demo_semantic_navigation.py \
 --source "$source_episode" --output "$output_run/navigation" --preserve-initial-state \
 --target bowl --navigation-mode hybrid --object-map hydra --perception daaam \
 --explorer frontier --goal-room "${GOAL_ROOM:-}" --vlm-device "${VLM_DEVICE:-cuda:1}" \
 --ring-checkpoint /nobackup2/le/.cache/hub/models--AinazEftekhar--OneRING/snapshots/adf715b75c11ca789a033c9f157360f50c48a6e0/ring_model_step_40356421.ckpt \
 --seconds "${NAV_SECONDS:-180}"
python3 - "$output_run/navigation/report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
# Docking is only permission to plan: M2T2 validates the full arm path before moving.
if not r.get('sensor_navigation_success') or not (r.get('manipulation_ready') or r.get('docking_success')) or not r.get('selected_object'):
 raise SystemExit('No verified sensor docking/target; no M2T2 execution.')
print('Sensor docking accepted for M2T2 planning; grasp success remains unverified.')
PY
bash scripts/test_m2t2_pick.sh "$output_run/navigation" "$output_run/pick"
printf 'First pick qualification completed. Carry/place/full-task scoring are not implemented in this runner.\n'
