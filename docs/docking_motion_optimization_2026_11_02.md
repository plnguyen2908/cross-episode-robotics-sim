# Observed-space docking motion optimization — November 2, 2026

The slow long approaches were split into 16–18 separate 25 cm advances. Each endpoint triggered braking and acceleration despite zero clearance stops in those baseline cases. More interpolation samples alone would not remove those endpoints.

## Implementation

`ShapedDocking.plan_advance` now selects up to 1 m of swept-clear observed floor, falling back to 0.75, 0.50, 0.25, 0.15 or 0.08 m when necessary. The distance budget leaves 1.3 m for precise docking. `smooth_docking` fits a collision-checked forward cubic Bézier curve; the executor follows it with commands at 50 Hz. The 0.30 m/s speed limit, acceleration limits, online clearance checks, camera geometry, final braking and alignment criteria are unchanged.

This is fewer stop/start segments, not fully continuous replanning: each selected horizon still has an endpoint. Unknown floor and intermediate obstacles reject longer advances. Execution uses onboard RGB-D and estimated SLAM state, not simulator poses or maps.

## Measured fresh-reset results

Baseline: `runs/docking_distance_suite7`. New: `runs/docking_distance_smooth1`. One two-room scene, four distances and two initial headings. All durations are simulated seconds.

| Case | Before (s) | After (s) | Saved | Advances before → after |
|---|---:|---:|---:|---:|
| close_forward | 7.52 | 7.52 | 0.0% | 0 → 0 |
| far_forward | 15.44 | 15.44 | 0.0% | 0 → 0 |
| farther_forward | 23.44 | 15.24 | 35.0% | 9 → 2 |
| farthest_forward | 43.94 | 32.34 | 26.4% | 16 → 4 |
| close_turned | 5.14 | 5.14 | 0.0% | 0 → 0 |
| far_turned | 9.44 | 10.14 | -7.4% | 2 → 1 |
| farther_turned | 22.74 | 18.34 | 19.3% | 9 → 3 |
| farthest_turned | 40.54 | 27.14 | 33.1% | 18 → 5 |

8/8 docking and independent frontal-diagnostic passes, zero audited contacts; maximum observed counter-normal error 3.07°. 120 regression tests pass, including new tests that reject unknown floor and intermediate obstacles on longer advances. All eight video durations match simulation time within 0.21 s.

The longer cases improve by 19–35%; far_turned increases by 0.70 s. This is a single-scene comparison, not a statistically repeated benchmark. Close cases retain their precise observation turns. Final braking, alignment and fresh mapping remain necessary; no contact checks were omitted. Long roughly-five-metre routes exceed the 5–15 s target because travel alone at 0.30 m/s needs about 17 s, before search, turns and precision docking. No fixed compression multiplier or video acceleration was applied.

## Videos and scope

[Side-by-side: farther forward](../runs/docking_distance_smooth1/comparison_farther_forward.mp4) — both play at 1×; the completed right-hand video holds its last frame, explicitly labeled.

[All eight videos and results](../runs/docking_distance_smooth1/README.md). Policy sources and fingerprints are saved in the run folder.

No grasp was executed. Four cases still require fresh final visual confirmation; the other four still require arm-path/grasp validation. None is reported as manipulation-ready. No merge to main was performed.
