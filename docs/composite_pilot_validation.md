# Composite pilot physical qualification — 2026-10-14

These are real persistent-scene rollouts using the fixed atomic interface. A scene
build or a successful standalone atomic demo is not a complete composite success.

| Task | Evidence | State |
| --- | --- | --- |
| Set table | [Report](../runs/composite_pilot_wn8qt5sf/set_table_01_original/report.json), [video](../runs/composite_pilot_wn8qt5sf/set_table_01_original/composite.mp4), [calls](../runs/composite_pilot_wn8qt5sf/set_table_01_original/skill_calls.json) | PASS: 24 calls, three bowl/cup settings, extras unchanged, empty hand. Video 941.72 seconds; total generation 1700 seconds. |
| Warm two pots | [Report](../runs/composite_pilot_3igr_1ib/warm_two_pots_01_original/report.json), [video](../runs/composite_pilot_3igr_1ib/warm_two_pots_01_original/composite.mp4) | PASS: 17 calls, both pots heated for the required duration on separate burners, both burners off, empty hand. Video 304.48 seconds; generation 869 seconds. |
| Prepare and serve meal | [Log](../runs/composite_pilot_ykom3o_5/prepare_and_serve_meal_01_original/run.log) | Still unqualified. The 65-degree door opening and food-container pickup passed. Placement fails native-mesh wrist clearance. Testing upright placement orientations from the saved failure state. |
| Sort groceries | [Report](../runs/composite_pilot_k2tivesx/unpack_and_sort_groceries_01_original/report.json), [video](../runs/composite_pilot_k2tivesx/unpack_and_sort_groceries_01_original/composite.mp4) | PASS: 24 calls; drinks and snacks remain inside their respective baskets, both baskets on the dining table, extras unchanged, empty hand. Simulation 486.83 seconds; total generation 954.71 seconds (15:55). |
| Clear and store | [Log](../runs/composite_pilot_jn0a8sqq/clear_and_store_01_original/run.log) | Still unqualified. The 65-degree opening and bowl pickup passed, but shelf placement failed. Testing upright placement orientations from the saved failure state. |

## Fixes made during qualification

- A shared persistent physical environment replaces the blocked factory stub.
- Top-down annotation checks reject assets qualified only for side grasps; saucepan
  lids are not selected as pots. Storage uses compact bowls.
- The microwave is placed on a clear countertop or an actual kitchen appliance cart,
  with collision geometry and source-object approach clearance.
- Replacement stove roots are aligned to the floor. Burner placement uses the
  supporting surface nearest the authored burner site, not an internal slab.
- Cabinet docking accounts for the opened door. Drawer binding uses the middle
  drawer, whose handle permits a top-down approach.
- Box collision faces participate in the finger contact-window test; selecting only
  mesh vertices incorrectly discarded valid snack-box and basket grasps.
- Atomic placement cannot secretly navigate. Compartment approaches enter through
  the open front. Failure remains a failed call, never a fabricated success.
- Initial state uses the actual compiled `model.mjb`; `mj_saveLastXML` could save an
  unrelated XML import after a model was constructed with `MjSpec`.
- Setup checks initial robot clearance. Stage changes stream to the terminal;
  failures retain error details and timeouts terminate child processes.

Earlier failed-run outputs are retained for diagnosis. The successful table run
predates the compiled-model snapshot correction: its `model.xml` is not a replay
model; use its authored scene, manifest and controller setup to reconstruct it.
New runs save `model.mjb` and `initial_state.npz` together.

The expanded focused regression run passed 168 tests; subsequent asset/IK/docking checks also passed. All ten scene-only checks (alternate layout and combined variation for each task) now build successfully. Scene builds do not establish manipulation success.

Remaining qualification: complete the two unfinished task rollouts, verify the alternative procedures, and run all seven variants of each task. Filled-basket transport passed in the original grocery rollout. The full 35-demo batch is not yet qualified.

## Additional qualification fixes

- Placement footprints use the intended upright object pose, not its tilted carry pose. Baskets pack from one side to preserve room for later objects.
- A relocated basket docks relative to its measured supporting table.
- Basket contents participate in payload collision planning. Only scratch states project them with the basket; live objects retain free joints and contact/friction dynamics.
- A stationary carry fallback collision-checks the measured pose without an unnecessary IK solve.
- Compartment alignment and insertion are checked before physical motion. The native-mesh fallback retains dense robot, payload and environment collision checks.
- Hinge grasps are screened for IK feasibility throughout the requested swing, before selecting a handle grasp. Execution still checks each physical stroke.
- Pickup approach configuration is separate from navigation, so a public pick does not depend on navigation having configured its grasp search.

Loaded-basket diagnostic evidence: [result](../runs/composite_place_debug_yb887339/debug_result.json). This is a replay from a failed state, not a training demonstration.

The alternative grocery procedure also passed a full rollout:
[report](../runs/composite_pilot_r6gzdawd/unpack_and_sort_groceries_02_alternative_procedure/report.json),
[video](../runs/composite_pilot_r6gzdawd/unpack_and_sort_groceries_02_alternative_procedure/composite.mp4).
It moves both empty baskets to the dining table before filling them: 24 atomic calls,
all measured goals passed, total generation 1317 seconds (21:57). Rendering overlapped
with placement diagnostics on the same GPU, so this is not an isolated timing comparison.
The original and alternative grocery runs have identical initial snapshots and saved
simulator state vectors (maximum absolute difference 0), establishing the shared
starting state for this procedure comparison.

The latest broad regression selection passed 133 tests. Compartment placement now
considers four yaw orientations, with upright and tilted releases for empty bowls
and closed rigid containers. It aligns outside the measured open door and held-object
footprint, approaching from the side opposite the hinge. Collision checks remain intact.
The inherited duplicate tabletop release-fallback pass is disabled for compartments;
the search checks a 120-second wall-time budget between candidate attempts.

Microwave placement now passes in a saved-state replay, including release, settling,
native-mesh checked empty-hand withdrawal, and measured containment:
[diagnostic result](../runs/composite_place_debug_b10b06ho/debug_result.json).
This is a diagnostic, not a complete meal demonstration. A fresh full meal rollout
is being qualified in `runs/composite_pilot_syxvovaq`.

## Two-object cabinet qualification (2026-10-14)

Before attempting six stored objects, the focused test opens the wider cabinet,
puts `bowl_01` and `cup_01` inside, and closes it (12 public atomic calls).
It retains the original scene and leaves the other objects untouched. It does
not change the clear-and-store recipe, which still assigns cups to the drawer.

```bash
ROBOCASA_DIR="$PWD/external/robocasa/robocasa" python scripts/qualify_composite_fixture.py cabinet_storage
```

The first attempt [report](../runs/qualify_cabinet_storage_gojfw4a5/report.json)
failed at door opening, before either placement. The passive door moved during
the free-space reach, invalidating the cached handle approach; native collision
checking caught gripper-base contact with the handle. The controller now refreshes
the grasp plan if the door drifts by more than one degree, with at most two
refreshes. The identical-scene retry `runs/qualify_cabinet_storage_34zgqojz`
avoided that stale approach but encountered handle contact with the gripper's
spring linkage during a subsequent stroke. Intermediate waypoints already passed
while redocking are now skipped, avoiding unintended reversal. The final retry
[report](../runs/qualify_cabinet_storage_d6wkr9zu/report.json) confirmed this skip
at 44.86 degrees, but still failed on spring-link/handle collision before loading.
None of these attempts reached either object's placement. The wider cabinet's
handle grasp/swing clearance remains unresolved; neither two-object loading nor
six-object capacity is qualified. Door regression tests passed (17 tests), and
the updated stroke tests passed (7 tests, including drift in both directions).

### Subsequent clearance and placement fixes

The swing cause was identified and fixed in scene setup; see
[cabinet drift inspection](cabinet_drift_inspection.md). The corrected run
`runs/qualify_cabinet_storage_2_928zdl` opened the door to 89.3 degrees and stored
the bowl, then failed cup placement at call 10/12. The door issue is resolved;
the initial two-object rollout did not pass.

The cup's placement succeeded in a saved-state diagnostic after changing the
fully open, laterally clear door approach to the measured compartment mouth.
The former approach forced a lateral detour and a 30 cm outward offset, creating
unreachable or blocked arm paths. Native robot/payload swept collision checks
remain required. [Placement result](../runs/composite_place_debug_nxicpk3g/debug_result.json).
This diagnostic starts with the bowl already stored and the cup physically held;
it is not a fresh end-to-end demonstration. Closing reached 0.2 degrees but the
global planner rejected the empty-hand withdrawal. A local IK withdrawal fallback
now retains native swept collision validation and passed at that saved failure
state. The [final diagnostic result](../runs/cabinet_release_replay_el1d53yv/result.json)
confirms both objects inside, empty gripper, and cabinet closed at 0.18 degrees.
The final regression selection passed 33 tests. A fresh combined 12-call test
is running; these saved-state successes do not replace that qualification.

The meal's missing snack support was traced to microwave-door contact knocking
the box off the counter. Datagen now reserves the door's full swing when placing
counter props. Run `runs/composite_pilot_qj08ecuo` subsequently passed opening,
loading and closing but failed Start because the snack box blocked the gripper
at the fixed control panel. Setup now also reserves a 15 cm clearance around
the appliance footprint for hand access. The fresh full rerun is
`runs/composite_pilot_85c3itu1`; the focused Start/Stop check is
`runs/qualify_microwave_51w6igj8`. The focused test passed all four calls and
rendered its [video](../runs/qualify_microwave_51w6igj8/composite.mp4).
The full meal remains in progress, not a qualified success.
