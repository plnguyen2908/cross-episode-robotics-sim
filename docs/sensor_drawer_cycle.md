# Sensor-only drawer cycle qualification

A complete, continuous **open → release → withdraw → regrasp → close → release → withdraw** passed in `runs/sensor_drawer_cycle_v6/report.json`. The fixture remained fully open after release and closed to within **0.091 mm** of its initial closed position; the final gripper was released. This is one prepositioned drawer component, not navigation or cross-scene qualification, and not a registered production skill handler.

## Execution interface and observations

`SensorDrawerMotion` (`cross_episode_sim/control/sensor_drawer.py`) accepts robot encoder/FK, gripper encoder, physical-command and observed-world planner callbacks. It never reads fixture coordinates, joint states, limits, contacts, or hidden scene labels. `move_until_resistance(opening=...)` advances along the observed panel normal and stops when measured robot tracking resistance persists. Its returned status is **motion resistance**, not privileged task success.

`scripts/test_sensor_drawer_cycle.py` continues the existing observed-handle opening harness in the same live simulator. That harness prints a private report; continuation never reads its success or fixture scores. Initial reset is only at episode start. A tracked handle pose comes from robot FK while holding the observed handle; no reference fixture pose is supplied. Private fixture state snapshots are used only in the final audit and never feed control.

Fresh onboard RGB-D provides collision obstacles. A 7.5 cm local intended-contact capsule around the TCP sweep permits handle interaction in the planner. All physical collisions stay enabled, and all other observed surfaces remain obstacles. The planner checks self collision, observed obstacles and densely sampled swept commands; it does not silently execute a rejected path.

## Fix and tests

Initial cycle trials failed during withdrawal despite physical release. The coarse observed voxel envelope falsely overlapped a proximal-arm collision sphere. In one case the raw observed clearance was 15.38 mm but the voxel clearance was −3.9 mm; a second case had 8.77 mm raw clearance and only 2.85 mm voxel clearance.

`sensor_drawer_geometry.py` now uses **2 mm voxels within 12 cm of calibrated robot geometry**, and 15 mm elsewhere. Collision margins and contact permission were unchanged. Occupancy merging uses sorted traversal rather than repeatedly searching an entire set for its minimum. It preserves exactly the occupied voxels without filling holes. The observed planner checks these boxes directly; no cuRobo fallback is enabled in this component test.

Four focused tests pass: observable resistance stop, blocked-plan nonexecution, exact occupied-voxel preservation, and adaptive resolution retaining every observation. Command:

```bash
.venv-sift/bin/python -m pytest -q \
  tests/test_sensor_drawer_motion.py tests/test_sensor_drawer_geometry.py
```

## Physical timing

Commands execute at 50 Hz with bounded quintic joint interpolation. No video acceleration, teleportation or disabled collisions. Encoder readiness determines gripper settling; persistent tracking resistance determines contact stops.

| Phase | Simulated seconds |
|---|---:|
| Approach, grasp, full opening and verification | 27.32 |
| Open-state release | 0.84 |
| Open-state withdrawal | 1.30 |
| Regrasp | 1.38 |
| Closing contact motion and endpoint settling | 17.64 |
| Closed-state release | 0.84 |
| Closed-state withdrawal | 1.16 |
| **Entire cycle** | **50.48** |

Open including release/withdraw is **29.46 s**; close including regrasp/release/withdraw is **21.02 s**. These exceed the 5–15-second target. The retained small contact increments and measured endpoint settling protect grip/contact tracking over the unusually long 60 cm drawer stroke. Further retiming remains an implementation gap; passing physics is not a claim that timing is fully optimized. No redundant base navigation or tuck/untuck was added.

## Artifacts and command

- Full cycle: [drawer_open_close_v6.mp4](/nobackup/le/cross-episode-robotics-sim/runs/navigation_previews/drawer_open_close_v6.mp4)
- Closing portion: `runs/sensor_drawer_cycle_v6/close_cycle.mp4`
- Reports: `runs/sensor_drawer_cycle_v6/report.json` and `opening/report.json`
- Live audit log: `runs/sensor_drawer_cycle_v6.log`
- Video was rendered from recorded physical states **after** execution, not replayed actions.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=.:external/DAAAM/src:/nobackup/le/cross-episode-robotics-sim/external/molmospaces \
OMP_NUM_THREADS=4 .venv-sift/bin/python scripts/test_sensor_drawer_cycle.py \
  --output runs/sensor_drawer_cycle_NEW
```
