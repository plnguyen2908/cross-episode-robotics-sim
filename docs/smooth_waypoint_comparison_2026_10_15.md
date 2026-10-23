# Smooth waypoint pilot: navigation and pick/place

Two matched runs passed all four public calls: navigate to food container, pick, carry to dining table, place. Initial saved simulator state, qpos, qvel, controls, and time were exactly identical. Both use the same datagen scene, asset, seed, nominal speeds, collision tolerances, grasp checks, and public interface.

- Baseline: `runs/smooth_baseline_tvn3yc28`
- Smooth: `runs/smooth_smooth_fwe_z21w`
- Metrics and A/B videos: `runs/smooth_waypoint_comparison_20261015`

## What changed

The opt-in `smooth_waypoint_experiment` controller flag enables two changes:

1. Join multiple existing A*/blended base segments using a cubic Hermite curve. Preserve the combined planned duration and endpoints. Validate the robot and payload sweep with the existing floor and native collision checks; fall back to the original route if validation fails. Reverse undocking stays unchanged. The cross-room carry's two segments were successfully joined into one; its curve validation took 3.08 wall seconds. The first empty approach already had one segment and was unchanged.
2. Interpolate the existing contact-IK joint waypoints at 20 ms instead of holding each target for 320 ms. Use shape-preserving slopes with zero endpoint velocities; recheck the interpolated robot and payload configurations against native meshes. Scale feedback updates with the sampling interval. Preserve the original target duration, endpoint validation, and grasp retention checks.

The contact interpolation was exercised in the two lifting phases. Existing cuRobo free-space approach and placement motions were retained. This is a continuity pilot, not a full replacement of every atomic trajectory, not OCBench's full randomized Cartesian generator, and not a change to the exported VLA action space. Default datagen does not enable the flag.

## Measurements

| Measurement | Current | Smooth |
|---|---:|---:|
| Successful public calls | 4/4 | 4/4 |
| Initial navigate, simulated seconds | 20.00 | 20.00 |
| Pick, simulated seconds | 18.034 | 18.354 |
| Carry navigation, simulated seconds (including preparation) | 74.60 | 74.52 |
| Place, simulated seconds | 24.026 | 24.026 |
| Total four calls, simulated seconds | 136.660 | 136.900 |
| Physical generation including setup, wall seconds | 177.08 | 182.93 |
| Cross-room base path, meters | 5.943 | 6.056 |
| Cross-room base RMS acceleration, m/s² | 0.02043 | 0.01402 |
| Cross-room peak base speed, m/s | 0.2247 | 0.1689 |
| Main lift RMS joint acceleration, rad/s² | 1.0482 | 0.05705 |
| Initial vertical lift RMS joint acceleration, rad/s² | 1.2097 | 0.1694 |
| Maximum navigation payload slip, mm | 0.322 | 0.205 |
| Final navigation position error, mm | 0.0067 | 0.1718 |
| Navigation collision metric, m | 0 | 0 |

Acceleration estimates use finite differences of actual simulator positions recorded at 25 Hz, then the RMS Euclidean norm (seven arm joints for lift, base x/y for navigation). They are not commanded acceleration limits or human-reference comparisons. Base acceleration decreased about 31%; main-lift joint acceleration decreased about 95%. Smooth motion did not materially reduce simulated duration and added validation cost. Runs were on independently leased GPUs, so wall-time differences are descriptive rather than a controlled hardware benchmark.

The original external review cameras were partly occluded by walls. A separate replay renderer uses the saved physical traces and kitchen/dining wall cameras plus wrist camera. This affects review only, not executed physics or policy observations.

## Videos

Top: current. Bottom: smooth. Both are normal speed; the shorter clip freezes its last frame until the other completes.

- [Navigation A/B](../runs/smooth_waypoint_comparison_20261015/navigation_comparison.mp4): loaded cross-room navigation, including carry preparation.
- [Pick and place A/B](../runs/smooth_waypoint_comparison_20261015/pick_and_place_comparison.mp4): pick followed by place; the intervening carry is omitted from this clip.
- [Machine-readable metrics](../runs/smooth_waypoint_comparison_20261015/metrics.json)

## Reproduce

```bash
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant baseline
/home/leos/.venvs/sim/bin/python scripts/compare_smooth_waypoints.py --variant smooth
/home/leos/.venvs/sim/bin/python scripts/render_waypoint_review.py BASELINE_FOLDER SMOOTH_FOLDER
/home/leos/.venvs/sim/bin/python scripts/summarize_smooth_comparison.py BASELINE_FOLDER SMOOTH_FOLDER --output COMPARISON_FOLDER
```

The experiment defaults to the saved successful meal scene at `runs/composite_pilot_wm1meqfa/prepare_and_serve_meal_01_original`; `--source` selects another compatible meal scene. It is not a standalone scene generator. Each physical run chooses an idle GPU and creates a unique directory.

Validation: 25 tests passed across `test_smooth_waypoints`, `test_shared_manipulation`, `test_loaded_carry`, and `test_pregrasp_standoff`; both physical runs passed. A single paired example does not establish robustness across layouts, speed settings, or objects. Next validation should separate the navigation and arm interventions and expand seed coverage before enabling defaults or adding randomness.

School Drive copies (verified after upload): [Navigation](https://drive.google.com/file/d/1OrzW_w4zgo6hTx60GLqAxAJqbDrGFeVA/view) · [Pick and place](https://drive.google.com/file/d/1YdA9Ppd6LTUFy7fcqMWs0evZsUZKHe2v/view).
