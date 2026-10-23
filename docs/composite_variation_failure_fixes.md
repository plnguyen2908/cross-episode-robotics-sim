# Variation batch failure investigation

Source batch: runs/composite_pilot_ntyu1n_9. It completed 23/35 successfully.
The 12 failures remain recorded; the original batch is never overwritten.

## Findings and changes under physical verification

- Pot original/alternate procedure: final right-knob release withdrawal cannot
  find sufficiently accurate IK. Composite recovery now issues explicit
  navigate(dock=...) then rotate_knob(state="off"), preserving the goal.
- Pot layout 3: front-control stove replacement discarded the native Euler
  rotation. Preserve the authored orientation (Euler, quaternion, or other
  supported MuJoCo encoding). Unit regression covers Euler and quaternion.
- Rotation-only furniture: rotating the sofa beside the wall violated room
  bounds. The preset now rotates the two freestanding armchairs instead.
- Grocery and cabinet object variants: policy previously only recovered picks.
  It now explicitly carries the same object to alternate placement docks before
  retrying place. Basket placement additionally considers upright quarter-turn
  orientations; containment does not require the source object's yaw.
- Alternate pot/container pickup: add physically checked higher rim candidates;
  retain original candidates, contact checks, lift and retention requirements.
- Meal layout: if loaded tuck *planning* exhausts its candidates, retain the
  measured grasp only if a complete loaded route passes native swept checks.
  Execution/tracking failures are not swallowed.
- Clearing layout: check alternative fixture approach docks, including lateral
  offsets for corner cabinets. Default dock remains first.
- Clearing closed-door shortcut: checked reverse departure before a tight
  same-room turn; check alternative local door docks before moving. Reuse the
  existing faster forward door-redocking executor.
- Cross-room escape: check straight backward/forward prefixes as well as lateral
  prefixes, accepting only a complete swept-clear route.

No collision tolerance, success condition, heating duration, or object pose is
changed during execution. The new furniture preset and corrected stove transform
are scene-generation changes, not runtime corrections.

## Verification state

Physical reruns are in progress. Some first attempts still failed; the added
dock candidates alone did not fix layout-3 stove navigation (the missing Euler
rotation was identified afterward). Do not interpret these changes as 35/35
qualification.

Rerun logs are in /tmp:
pots_knob_recovery.log, grocery_place_recovery.log,
clear_objects_recovery.log, pots_objects_rim.log, meal_objects_rim.log,
meal_layout_recovery.log, pots_layout_orientation.log,
clear_layout_lateral.log, clear_door_docks.log.
Each log starts with its unique OUTPUT path.

## Subsequent findings

- Baseline pots rerun m6o61vgi passed all goals and rendered its video.
- Correcting the stove rotation passed layout-3 navigation but exposed placement
  reachability. Burner targets now allow upright quarter-turn object orientations,
  just like basket containment; no tilt is added for burners.
- Food-container variant passed pickup and microwave placement with the rim
  candidates; full meal verification remains pending.
- Alternate-bowl clearing passed both cabinet placements; full episode pending.
- Pot057's raw mesh is about 40 cm across, but its *scaled simulation geometry*
  fits the front-burner spacing. The initial raw-mesh size hypothesis was wrong.
  A measured burner-footprint asset filter was added, but it does not exclude
  Pot057 and must not be described as fixing that grasp failure.
- Higher rim candidates also need to be allowed by the composite approach filter:
  a separate rim-search flag now permits their tilted fallback poses while
  preserving top-first ranking and rejecting upward approaches.

## Latest physical verification

The following reruns have passed all composite goal checks. Rendering is separate
from physical success; meal-object and stove-layout rendering is still pending.

| Original failure | Successful rerun directory under `runs/` |
| --- | --- |
| Pots, original | `composite_pilot_m6o61vgi/warm_two_pots_01_original` |
| Clearing, alternate objects | `composite_pilot_dc7c07is/clear_and_store_05_object_instances` |
| Pots, rotated furniture | See `/tmp/pots_rotation_fixed.log` OUTPUT |
| Meal, alternate objects | `composite_pilot_x2i0uguu/prepare_and_serve_meal_05_object_instances` |
| Pots, kitchen layout | `composite_pilot__wkmy03c/warm_two_pots_03_kitchen_layout` |

Remaining tests are not qualified yet. Do not launch a replacement full batch on
an assumption that every failure is resolved.

Additional fixes being verified:

- Basket entry uses native-mesh checked alignment above the rim and descent into
  the cavity. Conservative cuboid obstacles can incorrectly fill that cavity.
  Both drink placements passed; the full grocery task is still running.
- A corner cabinet can retain its already reached public navigation dock when
  every preset local door dock fails planning. The handle approach still needs
  full collision checks; this does not bypass a failed executed movement.
- Clearing from the dining side opens the right room-door leaf first, avoiding
  the closed right leaf blocking the left-leaf handle approach.
- A supported object's initial lift IK failure now rejects that grasp and tries
  another annotation. An airborne object is never released for this recovery.
  The alternate pot has not yet passed; its trace shows loss of contact during
  lift-clearance correction, rather than a simple missing grasp annotation.

The saved-pose fridge test is diagnostic only. It cannot qualify a complete meal
episode; a fresh full run is required after any successful local diagnostic.

- Alternate-order pots also passed all goals and finished rendering:
  `runs/composite_pilot_w_x33mn6/warm_two_pots_02_alternative_procedure`.
- Layout-3 clearing now passes cabinet opening and proceeds to bowl delivery;
  this is progress, not yet a complete-task pass.
- Oracle pick now permits up to three **distinct** grasp trials, preserving the
  rejected grasp set outside checkpoint restoration. Failed trials remain in
  `rejected_trials`; successful branches still require physical qualification.
  No new trial is launched if no grasp was selected or no new rejection exists.
- Compartment alignment can widen the IK search trust region from 2 to 6 radians
  after the restricted solve fails. Joint bounds and native swept collision
  checks remain enforced; a wider search is not a relaxed success tolerance.
- Door regrasp preflight failures can try three checked local docks while
  retaining completed hinge progress. Executed contact/tracking failures still
  propagate. This remains under physical verification on the fridge.

## Current summary

Latest focused regression suite: **92 passed**, with one upstream Pydantic
warning (`/tmp/final_variation_regression.log`). Unit tests do not replace physical
qualification. Six original failure configurations have successful full reruns;
six remain unresolved or under verification.

Latest active test logs:

- `/tmp/clear_layout_current_dock.log`: corner-cabinet clearing.
- `/tmp/clear_door_right_first.log`: closed-room-door clearing.
- `/tmp/pots_distinct_grasps.log`: alternate pot, distinct checkpoint grasp trials.
- `/tmp/pots_combined_lift_retry.log`: combined pot variation; check the worker's
  launch time before attributing newer code changes to this run.
- `/tmp/grocery_alignment_retry.log`: basket native-mesh entry plus wider IK search.
- `/tmp/fridge_local_regrasp.log`: saved-pose fridge diagnostic with local regrasp;
  still requires a subsequent fresh full meal run.

Completed rendered examples:

- [Stove layout](../runs/composite_pilot__wkmy03c/warm_two_pots_03_kitchen_layout/composite.mp4)
- [Alternate pot order](../runs/composite_pilot_w_x33mn6/warm_two_pots_02_alternative_procedure/composite.mp4)
- [Meal with alternate container](../runs/composite_pilot_x2i0uguu/prepare_and_serve_meal_05_object_instances/composite.mp4)

Latest addendum: layout-3 clearing opened the cabinet but failed the subsequent
loaded route. Cabinet carry docks now include the opposite lateral side, retaining
all original docks first. A fresh full rerun is logged at
`/tmp/clear_layout_carry_docks.log`. Its geometry regression passed with the
household physics suite (23 tests). The fridge local-regrasp diagnostic passed its
old approximately 39-degree failure point and reached approximately 63 degrees;
this is not yet an endpoint or full-meal pass.

## Closed double-door grasp correction

The right-first rerun `composite_pilot_yaobx_av` failed before grasp execution:
the gripper pad/follower intersects the opposite closed leaf. Reordering alone
was not a fix. Room-door bindings now add handle approach yaw candidates of
±30 and ±45 degrees after the straight approach. The closing direction remains
horizontal across the vertical handle, and every candidate retains complete
reach, contact-approach and hinge-path checks. Other fixtures keep zero yaw by
default. No leaf collision or clearance threshold is disabled.

Verification: 32 door/route/drawer tests passed. Full rerun:
`runs/composite_pilot_pnel9tk6/clear_and_store_06_open_shortcut`, log
`/tmp/clear_door_angled_handles.log`; physical result pending.

Follow-up physical inspection showed the first handle grasp succeeds and opens
the right leaf to about 11.81 degrees; the collision occurs on the next grasp.
The panel-push candidate gate formerly required exactly 12 degrees and now allows
0.5 degree settling error. Push collision checks still rejected the direct path.
The double-door recovery dock also pointed toward the *other* leaf. It now uses
the measured vector from handle to its own hinge for the lateral offset.

Room-door-only changes: angled grasps and checked hinge-side push redocking after
a direct push is rejected. Other fixtures retain their existing direct-push
fallback. Negative-opening hinge continuation is now capped symmetrically at
45 degrees, matching positive-opening hinges.

Latest test: `runs/composite_pilot_yrwcx4d3/clear_and_store_06_open_shortcut`,
`/tmp/clear_door_hinge_side.log`. Earlier angle-only, gate-only, and generic-redock
experiments were explicitly interrupted as superseded, not marked successful.
Door regression after hinge-side correction: 23 passed; preceding door/push/route
suite: 50 passed. Full physical result remains pending.

The hinge-side experiment physically reached approximately 45 degrees but failed
empty-hand withdrawal: mandatory wrist unwinding exceeded the local IK region.
A wider checked search reduced the error but still failed. Release now tries a
straight retreat preserving the measured wrist orientation if the unwinding
pose fails IK before execution. Collision/tracking failures are not swallowed.

Latest full test: `runs/composite_pilot_ygi9ktxi/clear_and_store_06_open_shortcut`
(`/tmp/clear_door_straight_release.log`). Tests: 38 door/release tests passed.
The preceding wider-IK run was interrupted after its known failure, with an
`interrupted.json` explaining the supersession. No full closed-door success is
claimed at this point.

Straight release passed physically. The following tuck failed even under dense
native-mesh validation: the folding elbow intersects the adjacent closed leaf.
Room-door local navigation may now retain the released empty-arm posture when
*tuck planning* fails, but only through the existing complete swept-route check.
A blocked retained-arm route executes nothing. Held or articulating hands are
ineligible. Latest run: `composite_pilot_fx9_gupy/clear_and_store_06_open_shortcut`,
log `/tmp/clear_door_retained_arm.log`.

The other outstanding verification attempts (both alternate-pot runs, grocery
alignment, layout-3 cabinet carry, and fridge diagnostic) have finished with
failures. They are unresolved, not running and not qualified. The six previously
successful configurations remain the only six validated fixes from the original
12 failures.

## Larger-cabinet full rerun passed

`runs/composite_pilot_8ri1agki/clear_and_store_03_kitchen_layout` passed all
24 steps and final physical goals, with live motion optimization enabled. The
94 cm double-door cabinet received both bowls; cups were stored and compartments
closed. The rendered video is `composite.mp4` in that directory. This is the
seventh verified fix among the original twelve failures. Five remain unresolved:
closed-room-door clearing, two alternate-pot configurations, grocery object
variation, and meal layout/fridge interaction. Earlier entries above are historical.

### Fridge panel closing investigation (2026-10-16)

The failed layout's open room-door leaf obstructs the outside of the fridge
panel. Closing that leaf through an explicit `close_door` call clears the
workspace. It must not be hidden inside the fridge atomic.

A base-assisted outside-face push physically reached 37.6 degrees from 81.8
with the room leaf closed. Returning to the handle then failed navigation;
a single base push all the way closed failed collision preflight near the
freezer/gripper linkage. An arm-only panel push from the reached 37.6-degree
state succeeded, released contact, and left the fridge at 0.041 degrees.
No joint teleport or collision-mask change was used during execution.

- Closing diagnostic: `runs/fridge_final_arm_push_o39uneq6/diagnostic_report.json`
- [Final closing video](../runs/fridge_final_arm_push_o39uneq6/fridge_close.mp4)
- Reusable strategy: `cross_episode_sim/skills/fridge_door_motion.py`
- The slower opening/contact cycle passed in `runs/fridge_slow_cycle_els3foq8`
  (81.49 degrees open, 0.077 degrees closed), using an arm-only final push.
  This is distinct from the requested final push-to-handle strategy.

The requested handle handoff also passed separately: from the physically
pushed 37.6-degree state, grasp the handle **without redocking**, close, then
release. Final angle: 0.033 degrees, no remaining handle contacts.
[Handle-handoff video](../runs/fridge_push_handle_handoff_gpa35d39/fridge_close.mp4).
The public fridge `close_door` now uses a partial base push followed by this
handle stroke, instead of the failed generic handle-redocking routine.
82 relevant regression tests passed. The public `close_door(fridge)` call
then passed from 81.815 degrees to 0.033 degrees, including the partial push,
handle grasp, final closing and release, in 36.852 simulated seconds.
Evidence: `runs/fridge_public_push_handle_elqe5_lf/diagnostic_report.json`.
The nearby room leaf was already physically closed. This saved-state test
does not establish full meal-task success.

### Restructured kitchen entry (2026-10-16)

The native fridge was too close to the added kitchen–dining room door.
Newly generated houses now reserve a 1.8 m entry area by extending the kitchen
floor and both side walls and moving the dining partition/room outward.
Native appliance coordinates are preserved. The living-room connection is
rebuilt from the expanded shell. Both floor layers and both wall layers move
consistently. `native_layout_metadata.json` records the entry clearance.

Layout 3 was rebuilt and rendered at
`runs/fridge_layout_clearance_0zveeld9`. A geometric sweep sampled 19 fridge
angles against 19 room-door fractions (361 combinations): no fridge/room-door
penetration deeper than 1 mm. This is geometric validation, not a successful
robot rollout. Layout regression checks passed for 1, 2, 3, 4 and 9.
Existing saved scenes and videos retain their original geometry; generate a
new episode to use the restructured layout.

### Push-side double room door: handle-to-panel strategy

The old successful set-table preview was recorded with both
`optimize_motion_components=false` and `smooth_action=false`. It demonstrated
pull-side success, not optimized push-side motion.

The new public room-door opening branch detects the push side using the hinge
axis, handle position, requested angular direction and current base position.
It grasps from the existing approach and physically pushes the handle to about
15 degrees, releases, then pushes an inboard panel patch. The gripper roll is
mirrored for the hinge direction. The wrist follows the panel rotation; the
base supplies half of the translation and arm IK supplies the rest. This keeps
the base out of the door swing. A single local contact dock is allowed only
when direct panel preflight fails. Physical execution failures are propagated.

Defaults for other fixture motions remain unchanged. Selected contact settings
are scoped and restored. No fixture pose edits, disabled collisions or relaxed
tracking thresholds are used. The old repeated handle-stroke/redocking sequence
is bypassed on the push side. Logs label the handle phase as a push.

Public-call diagnostics from the failed clearing scene, with optimization on:
- Right leaf: `runs/room_push_public_cstcjwdo`, successful call 0, 28.678 simulated
  seconds (its subsequent left-leaf trial failed before the mirrored-roll fix).
- Left leaf after mirroring: `runs/room_push_left_i9o2p1g8`, successful call,
  29.148 simulated seconds. Both started from physically reached saved states.
- 70 door/fixture regression tests passed.

The uninterrupted public-call sequence passed in
`runs/room_push_public_wji4gl35`: open right leaf (28.678 simulated seconds),
navigate to left leaf (11.700 seconds), open left leaf (29.184 seconds).
Both openings verified their physical endpoint after release. It starts from
the failed clearing scene's saved post-navigation state; there is no state
reload between these three calls. Video: `push_side_doors.mp4` in that folder,
normal simulation-time playback, with current smoothing enabled.
These atomic diagnostics do not establish success of the whole clearing
composite; no full composite rerun was started for this fix.

### Removed the push-side handle-to-panel tuck (2026-10-16)

`runs/room_push_public_332oagh5` passed the uninterrupted right-open, navigate,
left-open sequence. Each opening takes 24.9 simulated seconds, with zero
`tuck_for_navigation`/untuck events inside either open call. After releasing
the handle, the arm moves directly to the panel precontact pose, expressed for
the short upcoming base alignment. Local navigation checks the extended arm
instead of folding it. The panel push then uses the base and coordinated arm.
The separate navigation between leaves still uses the travel posture.

Video `no_tuck_openings.mp4` shows both openings at normal simulated speed,
with the between-leaf navigation omitted. 72 regression tests passed.

The full meal rerun with the enlarged entry (`composite_pilot_utkjtz2v`) failed
at call 5/29, carrying the food container toward the microwave: no collision-clear
loaded carry posture could be planned. It did not reach the fridge. The new
layout is geometrically checked but this meal rollout is not qualified.


### Stove and grocery placement follow-up (2026-10-17)

The combined stove variation now passes with two Pot_22 assets in layout 3,
including the furniture and procedure changes. The full run is
`runs/composite_pilot_2gg10n88/warm_two_pots_07_combined` (17/17 steps;
21m59s wall time including rendering). Both pots meet the heating goal and both
knobs finish off. The second placement required one redock.

[Stove video](../runs/composite_pilot_2gg10n88/warm_two_pots_07_combined/composite.mp4)

The placement controller preflights lower burner approach waypoints (8 cm,
4 cm, then direct) and additional upright cookware yaws. Every candidate still
checks the robot and held object against native meshes before executing.
This removes an unnecessary high waypoint that exhausted arm reach.

Grocery sorting is still under full-run validation. The larger purple/orange
baskets match the juice/snack colors. Basket placement now allows a lower
mesh-checked approach for short objects; tall objects retain overhead-first
alignment and corner-first packing. Applying the short-object ordering to all
objects made juice placement much slower, so that ordering was corrected before
continuing validation. An isolated saved-state snack placement passed, but that
is not a full-composite success. Latest full rerun:
`runs/composite_pilot_eqes4441/unpack_and_sort_groceries_05_object_instances`.

Validation: 87 tests passed across household physics setup, composite tasks,
and household assets. No collision thresholds or goal checks were relaxed.


Grocery follow-up: the `eqes4441` full run passed all four grocery placements,
then failed filled-basket pickup. Added ±20-degree exterior-tab grasp hypotheses;
these remain subject to normal physical grasp validation. The orange basket then
passed a saved-state pickup/carry/place diagnostic (`composite_place_debug_68j102l6`).
The purple basket passed pickup/carry but initially failed dining placement.

For basket alignment only, failed conservative IK checks now permit one additional
IK candidate whose entire interpolated path is checked against native robot,
basket, and contents geometry. Actual environment and self-collision thresholds
are unchanged. This production path passed purple-basket placement at alternate
dock 3 in the saved-state diagnostic (`/tmp/purple_basket_production_place.log`).
An extra reverse-yaw placement experiment did not fix the issue and was removed.

99 regression tests pass, including explicit rejection of both environment and
self-collisions in the basket fallback without stepping the live world. The full
rerun with these changes is `runs/composite_pilot_ixn3iowh/unpack_and_sort_groceries_05_object_instances`;
it passed all 24 steps and the final goal check (20m07s before rendering;
22m20s including rendering).


**Final full-run outcome: both remaining variations pass.** Grocery video:
[successful grocery sorting](../runs/composite_pilot_ixn3iowh/unpack_and_sort_groceries_05_object_instances/composite.mp4).
Its accepted trajectory is about 4m30s of simulated time. All four groceries remain
sorted, and both filled baskets finish on the dining table. No recovery dock was
needed in this final grocery rollout. These results qualify these two configurations;
they do not establish a success rate over all seeds/layouts.
