# Camera mount regression — October 21, 2026

The previously successful pot baseline and the failed new baseline have identical
existing geom sizes, positions, rotations and collision masks. The new model adds
the camera pole and housing. Joint names and ordering match.

Two problems were found:

- The original mount at base x=-0.20 m intersects the elbow during saved successful
  arm movements. Checking only the initial travel posture missed this.
- The generic eight-cell AABB approximation inflated the 12 mm-radius pole into
  89 mm-radius planner spheres, introducing additional false obstructions.

The mount now sits on a solid rear bracket at x=-0.40 m, centered behind the arm.
Height remains 1.70 m and downward tilt remains 30 degrees. Collision geometry is
retained. A conservative capsule sphere cover avoids the inflated approximation.
No task plan, success criterion, collision mask or video speed was relaxed.

## Verification

- Six geometry/camera/depth unit tests pass, including cylinder and end-cap
  coverage and rejection of the old artificial obstacle around the mast.
- All 5,200 saved poses of the old successful pot example clear the revised
  physical mount at the 0.5 mm penetration threshold. This checks geometry only.
- All 2,270 saved grocery poses also clear the revised mount.
- [Clearance reports](../runs/mast_regression_20261021/).
- Fresh baseline executions reuse the failed runs' authored scenes and configs;
  they restart physics and call the normal atomic interface. Set-table and warming
  two pots have passed their complete goal checks and rendered videos; the other
  three are still running. These five reruns use the rear bracket, before the plate revision.

| Task | Physical rerun |
|---|---|
| Set table, one person — passed | [run](../runs/regression_set_table_ylzgg04i/) |
| Clear and store, one person | [run](../runs/regression_clear_and_store_iihdpkei/) |
| Sort groceries | [run](../runs/regression_unpack_and_sort_groceries_7nb2ybkt/) |
| Warm two pots — passed | [run](../runs/regression_warm_two_pots_6obl6hb0/) |
| Prepare and serve meal | [run](../runs/regression_prepare_and_serve_meal_xqcn13r4/) |

Existing batch workers retain the robot model loaded at startup. Editing the
mount cannot repair an already-running process or its saved model.

Controlled rerun command (starts a fresh run, leaves the source untouched):

```bash
python scripts/retest_composite_scene.py --source runs/composite_pilot_23kw4435/set_table_01_original
```

## Forward navigation departure recovery

The rerun opened the cabinet but exposed a separate integration gap: the
cross-room dock-escape fallback was enabled only for holonomic `blended` mode,
while optimized demonstrations select forward-facing `turn_drive`/forward curves.
The fallback now supports forward-only operation with a swept-checked straight
reverse/forward prefix. Sideways departures remain unavailable in this mode.
The reverse prefix is retained through execution rather than reset after planning.

Nine navigation unit tests pass. From the saved failed cabinet-departure state,
the new navigate call succeeded: 0.20 m reverse departure, 5.91 m total travel,
14.02 simulated seconds, zero measured collision metric, and no holonomic travel.
[Diagnostic result](../runs/retry_cabinet_departure_vr_4qaj2/result.json).
The complete clear-and-store rerun above is still required for whole-task success.

## Rear extension plate

The user requested a plate to support the rear pole. The default mount now uses
an approximately 0.318 kg, 245 × 80 × 6 mm solid plate, spanning base x=-0.425 to
-0.180 m. It overlaps the rear edge of the 481 mm base; the pole remains centered
at x=-0.400 m. Height and tilt are unchanged. The wider initial prototype clipped
the stove on a recorded turn and was narrowed before acceptance.

[Robot/mount preview](../runs/rear_camera_plate/mount.png) is an external diagnostic
view with room clutter hidden for visibility, not a policy observation. Physics
collision geometry remains enabled. All 5,200 old successful pot poses clear this
plate ([report](../runs/rear_camera_plate/clearance.json)). A fresh set-table physics
run with the plate is in progress; earlier baseline reruns used the narrow bracket.

## Clear-and-store final drawer close (October 21)

The new batch `composite_pilot_b5yttahs/clear_and_store_01_original` completed
15 calls and failed on `close_drawer`. All 146 recorded failed-call poses had
zero mount contacts. The right finger progressively unloaded during the base
push; the drawer was still about 0.396 m open at failure. This rules out direct
mount obstruction for this call, not every possible inertial effect of mounting.

The shared drawer executor now closes at up to 0.06 m/s (opening remains up to
0.12 m/s) and uses a 2 mm settling target instead of 5 mm. Settling remains bounded
and feedback-driven. Bilateral contact is still required during the stroke.
At the final commanded waypoint only, reaching within 1 mm of the closed stop
permits a finger to unload before release. This is stricter than the unchanged
public closed-position tolerance of 10 mm; it does not permit an incomplete
stroke to pass. Collision checking remains enabled.

Saved-state diagnostics reconstruct recorded qpos/controls with zero initial
velocity in the original authored scene and current rear-plate robot. They are
phase regressions, not fresh full-composite runs. The original recorded grasp
now reaches closed within 0.07 mm:
[stroke result](../runs/retry_drawer_close_ilmux2h1/result.json).
The complete public-skill check is saved separately at
[close skill result](../runs/retry_drawer_close_ajpk6u21/result.json).

Resume keeps successful examples and regenerates failed/unfinished examples;
it does not resume physics at the failed atomic call:

```bash
bash scripts/generate_composite_variations.sh --resume runs/composite_pilot_b5yttahs --workers 5
```

Final validation: the public `close_drawer` call passed its measured postcondition,
including release and withdrawal, in 17.206 simulated seconds. This exceeds the
15-second target to preserve contact reliability. The drawer finished within
0.017 mm of closed after release. All 35 focused fixture/contact-completion tests
passed. No claim of a fresh full 35-example batch validation is made.
