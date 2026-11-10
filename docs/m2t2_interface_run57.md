# M2T2 skill-interface physical qualification — run57

**Result: full set-table success**, from the initial scene with no continuation. Two bowls and two cups were served. Final semantic checks passed (requested settings, no wrong table items, empty gripper). The legacy scorer key is named `three_complete_settings`, but the configured request was two people.

- Public skill calls:17/17 successful (`explore`, then four `navigate → pick → navigate → place` sequences).
- Backend stages:33/33 successful.
- Wall time:2310.88s (38m31s), before offline rendering.
- Offline rendering:444.01s (7m24s);33/33 recordings rendered without errors.
- Full video:551.2s (9m11s), original10fps simulated playback; no time acceleration.
- Recorded simulated motion:548.22s across stage traces; boundary sampling means this differs slightly from total video duration.
- Runtime:zero checkpoint restores, one renderer initialization.
- Regression suite:51 tests passed.

## What changed

The interface delegates to the existing stage functions in `sensor_set_table.py`; it does not implement a second navigation or grasp policy. Concrete dispatch tests verify each transfer uses its own pick/carry/placement inputs and the next transit starts at the previous return. The oracle interface remains separate and unchanged.

Three low-level failures were diagnosed and corrected during qualification:

1. False old-keyframe revisits shifted the tracked pose sideways. Revisit candidates now require agreement with consecutive RGB-D registration.
2. Grasp insertion excluded the target from all collision checks. The observed target is now checked against open-gripper linkages while allowing pad contact. The recorded failed insertion is rejected; the three preceding successful insertions pass this check.
3. Undocking chased an exact endpoint despite reaching safe transit space. It now brakes in the validated interior of the free transit region, retaining swept clearance checks.

Execution uses sensor estimates and calibrated robot geometry. Private saved physics/contact data were used only for diagnosis and final scoring. No contacts, collision geometry or task-success rules were disabled.

## Scope and remaining gaps

This is one successful development example, not a held-out success-rate estimate. Run55 and run56 failed after three completed transfers; their logs remain preserved. Fixture skills are not yet registered in this continuous backend. Navigation/placement docking and two cup grasps exceed the typical5–15s target; long room routes, alignment, contact phases and retained posture transitions remain visible in the timings below. Current return stages explicitly retain a checked navigation posture because pickup navigation does not yet support posture retention. Do not claim all human-like motion defaults are fully satisfied.

## Artifacts

- Run directory: `runs/set_table_pipeline_development/full_attempt_57_departure_region`.
- [Full video](/nobackup/le/cross-episode-robotics-sim/runs/navigation_previews/full57_m2t2_interface_success.mp4).
- [Task report](/nobackup/le/cross-episode-robotics-sim/runs/navigation_previews/full57_report.json).
- [Public skill journal](/nobackup/le/cross-episode-robotics-sim/runs/navigation_previews/full57_skill_calls.json).
- [Live/stage log](/nobackup/le/cross-episode-robotics-sim/runs/navigation_previews/full57_m2t2_interface_live.log).
- Source snapshot: `runs/set_table_pipeline_development/full_attempt_57_source`.

## Stage timings

| Stage | Wall seconds | Recorded simulated seconds | Result |
|---|---:|---:|---|
| explore_dining_destination | 264.5 | 33.5 | Passed |
| 1_transit_kitchen | 86.2 | 43.5 | Passed |
| 1_navigate_0 | 65.7 | 9.8 | Passed |
| 1_pick | 28.1 | 7.2 | Passed |
| 1_carry | 100.1 | 38.1 | Passed |
| 1_observe_table | 5.7 | 0.0 | Passed |
| 1_place_dock | 51.1 | 12.4 | Passed |
| 1_place | 37.0 | 7.2 | Passed |
| 1_return | 11.7 | 4.5 | Passed |
| 2_transit_kitchen | 90.0 | 40.2 | Passed |
| 2_navigate_0 | 59.3 | 7.5 | Passed |
| 2_pick | 39.9 | 8.0 | Passed |
| 2_carry | 105.0 | 33.4 | Passed |
| 2_observe_table | 5.7 | 0.0 | Passed |
| 2_place_dock | 74.7 | 18.4 | Passed |
| 2_place | 44.3 | 7.5 | Passed |
| 2_return | 11.7 | 4.8 | Passed |
| 3_transit_kitchen | 101.2 | 41.6 | Passed |
| 3_navigate_0 | 53.0 | 5.9 | Passed |
| 3_pick | 142.7 | 18.3 | Passed |
| 3_carry | 114.0 | 33.5 | Passed |
| 3_observe_table | 6.2 | 0.0 | Passed |
| 3_place_dock | 78.1 | 19.1 | Passed |
| 3_place | 34.1 | 8.2 | Passed |
| 3_return | 13.1 | 3.7 | Passed |
| 4_transit_kitchen | 140.6 | 50.6 | Passed |
| 4_navigate_0 | 56.7 | 8.5 | Passed |
| 4_pick | 92.4 | 17.1 | Passed |
| 4_carry | 136.2 | 32.3 | Passed |
| 4_observe_table | 3.9 | 0.0 | Passed |
| 4_place_dock | 74.4 | 19.5 | Passed |
| 4_place | 170.9 | 10.0 | Passed |
| 4_return | 11.9 | 3.6 | Passed |
