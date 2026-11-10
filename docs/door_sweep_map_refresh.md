# Door sweep map refresh

Opening a door changes the occupancy map. Integrating every intermediate door
pose as a static obstacle leaves a wedge across the aperture. Sparse depth rays
may not revisit every stale voxel after the door opens.

`ObservedNavigationMap.invalidate_door_sweep()` uses the sensor-estimated hinge,
closed handle and opened handle to expire accumulated occupied evidence inside
the swept region. It retains the hinge neighborhood, final leaf slab and cells
outside the sweep. The handle-to-edge allowance is 15 cm and the slab margin is
10 cm; these are model assumptions, not ground-truth fixture dimensions.
Expired cells become **unknown**, not free. Existing observed floor support is
retained. Fresh RGB-D endpoints restore current obstacles; free rays and the
projective depth filter establish clearance. This must not be represented as
proof that the robot can cross.

The live opening wrapper applies the update only after the observed opening
succeeds (at least 80 degrees). It does not consume private scoring fields.

## Validation

- Unit tests cover both sweep directions, frame/final-leaf preservation,
  unknown-space behavior, and fresh obstacle evidence.
- Saved-recording remapping is a diagnostic, not a physical rerun or the exact
  map used by the original execution.
- Physical crossing is pending user review of the offline animation. No new
  live execution is authorized by this validation status.

## Offline result, November 7

Saved recording: `closed_two_room_13_local_invalidation`. The update expired
4,652 occupied voxels; the solid orange wedge disappears in the replay.
23 map/portal tests pass. A point A* route exists through observed floor,
but its minimum floor clearance is only 5 cm. This is **not** a certified
robot-width route: unobserved floor gaps remain. Do not describe this as a
perfect crossing result or start the live test without the requested review.

Review animation: [door sweep refresh](../runs/set_table_pipeline_development/closed_two_room_13_local_invalidation/door_sweep_fixed_review.mp4).
Connectivity diagnostic: [map and point route](../runs/set_table_pipeline_development/closed_two_room_13_local_invalidation/door_filter_route_check.png).

## Exploration default

`LiveDoorOpener.open()` now calls the shared `refresh_opened_door()` after
a successful sensor-observed opening, for either hinge direction and each
opened leaf. This combines sweep invalidation with fresh local depth updates.
Exploration keeps its existing unfreeze/cross/refreeze lifecycle. Missing
accepted pose or hinge estimates cannot authorize geometric invalidation; the
fallback is depth-only refresh. The user authorized a new live exploration
validation after reviewing the offline animation.

### Live default validation: closed_two_room_15_sweep_default

The two-room initial-state exploration test completed unsuccessfully. Opening
succeeded at 89.28 degrees; the default invalidated 579 occupied voxels and
requested local depth refresh. The map unfroze as intended. Crossing then
rejected all 13 observed exit candidates with `unobserved_base_support`;
the observed floor does not cover the required robot footprint. This is not
a completed two-room exploration and must not be reported as successful.
The run used one visually left leaf, leaving the other closed, and did not
render video during execution.

## Crossing center: observed open segment

`navigation/open_aperture.py::observed_open_aperture` finds a bounded gap in
occupancy at the estimated doorway plane after fresh post-opening observations.
The two occupied boundaries (frame, wall, or closed leaf) determine its midpoint.
The hinge/handle estimate is only a search seed. The paired-door average is no
longer passed as a crossing target. Unbounded unknown space is not an opening;
missing observed boundaries produces `open_aperture_unobserved`.

The shared post-opening handoff resolves this opening by default. Exit candidates
lie on its centerline (within grid quantization), rather than a 1.8 m-wide lateral
search band. The approach and crossing face perpendicular to the doorway.
Existing robot-footprint and swept-motion checks remain enabled.

On run 15's saved map, the observed segment is 0.70 m wide, centered at
approximately (-3.468, -1.546) in the estimated map frame. This is a 7.5 cm
shift from the previous seed. This correction alone is not proof that floor
support gaps or full physical crossing are fixed. The revised policy has not
been run live yet. 35 map/aperture/portal tests pass, including single doors,
either double-door leaf, both open, rotated doorways, and missing boundaries.

## Exploratory crossing: remove pre-observed exit veto

User decision November 7: do not require the unexplored room's floor to be
mapped before crossing. `PortalCrossing` now proposes geometric centerline exits
0.55–0.85 m beyond the opening, without an observed-floor or room-transit-disk
prerequisite. Its shape checker skips the floor-support observation veto only
for this crossing operation; the map is not painted free. Known obstacle
collision checking, fresh-pose tracking, and physics remain enabled. Sensor
fusion continues during crossing so newly observed obstacles can stop motion.
This supersedes the earlier pre-observed-floor crossing requirement, not the
ordinary manipulation docking behavior. Physical success still needs testing.

## Runtime collision veto removed for crossing

Per user request, doorway crossing uses `runtime_collision_stop=False` in
`ObservedUndock`. Candidate selection and route planning retain observed
collision geometry, including door clearance. Once planned, execution no longer
rebuilds a collision checker and aborts on later map overlap. Physical collisions
remain enabled; collisions are not automatically classified as task failure.
Pose tracking, route following and timeout handling remain. This setting applies
to doorway crossing, not unrelated manipulation/undocking callers. Tests cover
continued execution when a later checker would reject, and the portal default.
A new physical crossing result is still pending.

## Run 18: physical doorway crossing passed

`closed_two_room_18_no_runtime_veto`: `opened_passage_ready`, 170.92 simulated
seconds / 953.84 wall seconds for opening plus crossing exploration. Private
diagnostics record dining-room entry (2.6 seconds sampled interior visitation)
and zero navigation contacts. The pipeline continued to dining-destination
exploration. This establishes crossing success, not full-house coverage or
set-table task completion. Speed remains an implementation gap.

## Post-opening crossing runtime optimization (pending physics comparison)

The successful run still fused navigation and wrist depth at 10 Hz throughout
crossing. Planned crossing now fuses navigation depth at a 0.5 s interval, or
earlier after 0.2 m translation / 0.35 rad turn; localization remains 10 Hz and
control 50 Hz. Wrist geometry fusion remains active for the immediate post-door
refresh, but stops after the crossing route is planned. Mapping resumes normally
outside that planned crossing. Consecutive lattice rotation waypoints collapse
only when a direct swept collision check passes. This is code/test validation,
not a measured runtime improvement yet; do not claim a faster physical result.

## November 7: opening recovery and station-memory handoff

Run 21 failed before crossing: the short pull moved the TCP but barely rotated
the observed panel, then `fit_held_panel` lost association. The previous code
aborted instead of trying its bounded opposite-direction probe. It also moved
the estimated handle contact cloud with the TCP before establishing a hinge,
which falsely turned the stationary handle into an obstacle on the return.
The contact cloud now stays at its observed position until hinge estimation;
panel-association loss can enter the existing reverse-probe recovery.

A saved-state component test showed that recovery alone was insufficient: the
original grasp did not move the door. Aligning the M2T2 proposal to the observed
panel and centering its jaws on the observed handle produced 60.4 degrees of
opening. That isolated test stopped at arm reach after floor-support-limited
base repositioning; it is not a successful full-opening qualification. Live
opening now uses these existing sensor-based grasp refinements. No fixture
truth is used to refine the grasp.

Closed-passage exploration now optionally records furniture landmarks during
its initial search and passes them to the next station-exploration stage. The
pipeline enables this: previously the next stage required a kitchen landmark
that the first stage had never saved. This avoids discarding useful semantic
observations before crossing.

Run 22 (`closed_two_room_22_centered_handoff`) passed closed-passage opening/crossing (475.0 wall seconds) and station discovery (45.9 wall seconds), 520.9 seconds total. The door reached 90.26 degrees, with no private navigation contacts. This validates the route and station-memory handoff, NOT 90% exploration of both rooms: execution stops at the dining station, and private floor coverage remains incomplete. Full set-table was not executed. The focused handle, station, follower,
portal, undock, scan and map-update tests passed (35 tests). Neither those tests
nor the isolated 60-degree opening establish full exploration success.

### Crossing speed correction (run 23 pending)

Run 22's sampled progress attributes about 59 wall seconds to the standalone
`face_open_doorway` phase and 94 seconds to the crossing follower. These are
interval estimates from progress samples, not exclusive profiling timers.
The map was already unfrozen: the extra delay came from a separate directed
view turn and the manipulation-departure speed profile (0.2 m/s, 0.3 rad/s).

The live handoff now uses two fresh sensor packets without a separate directed
view turn. Its shape-checked route uses the normal exploration limits (0.4 m/s,
0.6 rad/s, unchanged 0.4 m/s² and 0.8 rad/s² ramps). Generic manipulation
undocking retains its previous limits. Consecutive turns collapse only when
the full rotation is swept-clear. Map fusion is no longer restricted to the
near-door region once the estimated base reaches 40 cm before the crossing
plane; the new room is integrated while crossing. Localization remains online.

Twenty-four focused follower, undock, crossing, and update tests pass, including
same-route faster-motion completion and blocked-turn preservation. Run 23
`closed_two_room_23_exploration_speed` is the physical qualification; speedup
and success must not be inferred from configured limits alone. Like run 22,
this pipeline test ends at station discovery, not 90% whole-room coverage.

Run 23 completed successfully for the tested route + station-discovery scope.
Door opening reached 90.17 degrees. The handoff-to-crossing completion interval
fell from about 164 to 66 wall seconds (60% reduction), with no recorded private
navigation contacts. Sampled crossing-following motion fell from 47 to 27
simulated seconds (94 to 59 wall seconds). The door-opening endpoints and paths
vary slightly between live runs; these are end-to-end observations, not a
controlled identical-trajectory benchmark. New-room fusion was enabled before
the doorway plane, without the standalone directed-view turn. Ninety-percent
coverage of both rooms remains unqualified; this test ends at station discovery.

### Full exploration is not a doorway/station smoke test

The closed-door pipeline previously passed `--stop-after-crossing` and accepted
`passage_ready`, then stopped exploration after finding the dining station.
That did not meet the requested whole-room coverage target. The pipeline now
runs `explore_closed_rooms` without that flag and requires the coverage script's
`success` (`observed_topology_complete`) before station handoff. The explicit
standalone `--stop-after-crossing` option remains a component-test option only.

The supported-wall enclosure fallback also incorrectly filled across observed
interior walls and doorway partition seams. Subtracting those partitions from
the fallback preserves distinct room footprints. On run 23's saved observed
map, the corrected estimator gives approximately 96.7% first-room coverage and
45.4% second-room coverage, rather than merging the rooms. These are sensor
estimates, not simulator room boundaries. Added regression tests cover both
pipeline stopping behavior and fallback partition preservation; 23 focused
tests pass. Run 24 (`closed_two_room_24_full_coverage`) starts from the initial
closed-door scene and is pending physical qualification.

Run 24 stopped during crossing at the generic `OnboardBaseActuator` nominal
travel-posture tolerance (0.05 rad). Saved-state diagnostics also showed light
pad/door contact; this is not evidence of a completely contact-free route.
Planned portal motion now uses the same `PlannedMotionMonitor` contract as
fixed-map transit instead of the generic tuck veto. Route planning and physical
collisions remain unchanged; the monitor observes localization, observation
freshness and motion progress. Unplanned motion retains the original posture
requirement. Fifteen focused monitor/bridge/portal tests pass. Run 25
`closed_two_room_25_full_coverage` tests this from the initial state.

Run 25 reached `observed_topology_complete`: observed navigable-area coverage
98.84% / 90.27%, with no unresolved portal. The coverage stage took 743.5 wall
seconds including wrapper overhead (176.14 simulated seconds). It continued
after crossing and explored the second room. Private diagnostics reported
whole-room floor coverage of 89.16% kitchen / 90.33% dining; this denominator
differs from the policy's estimated navigable area. The contact-free physical
audit did NOT pass: gripper/door contact was recorded (maximum reported overlap
1.89 mm). Do not describe this as contact-free or 90% ground-truth coverage of
both room floors. The generic posture veto is removed from planned crossings;
physics and diagnostic contact recording remain enabled.
## Closed-door exploration to task-navigation handoff (run 26 investigation)

Run 26 completed both rooms and confirmed the kitchen and dining-table stations,
but the first return stopped before any motion (`plans: 0`,
`undock_route_unverified`). The station-label stage tried to seed the success
graph from its own short trajectory, which did not include the earlier kitchen
view or doorway crossing. Its report explicitly said `recorded: false`.

The fix preserves accepted SLAM poses in the continuous tracking session and
records the full exploration route. Both empty-hand transit and carry consult
that graph before the circular-clearance departure fallback. Graph routes retain
pose headings; proposed joins, turns and translations are checked against the
observed map with the current robot shape (including the observed payload for
carry). The larger exploration circle no longer rules out a previously crossed
narrow opening by itself. The map is frozen during task navigation; graph reuse
does not erase obstacles or replay control commands. Successful local approaches
are also recorded so manipulation docks can connect back to the graph.

The persistent map handoff now preserves its source voxel configuration as well
as its arrays. Interpreting 5 cm height voxels as 2.5 cm voxels would otherwise
move observed obstacles vertically in the carry/placement stages.

Validation: targeted tracking, route, orchestration and runtime tests passed
(44 tests); the narrow-door regression verifies that a robot-shaped route can
pass where the transit circle cannot, while a genuinely closed wall remains
blocked. A further 21 route/map-continuation tests passed after the voxel metadata
fix. Physical full-task validation is in progress. Run 27 opened the door but
stopped before the graph handoff: the crossing approach contacted the opened
leaf and pushed it partly shut. The diagnostic base geometry is unnamed, so
contact inspection must use body ancestry as well as geom names. The crossing
now first moves outward beyond the sensor-estimated leaf reach, aligns with the
opening, then crosses straight; all three legs retain swept checks. Run 28 tests
this from the original closed-door scene.

Full-coverage exploration also exports its already confirmed dining-table
landmark as a navigation target. When both kitchen and table are available, the
pipeline proceeds directly to task navigation: no second station-search stage,
extra table approach, or repeated labeling. The missing-station fallback remains.
Fresh local tabletop perception later serves physical placement, not repeated
room/station discovery.

Run 28's outward-first route avoided the leaf but timed out while repeatedly
recovering the same turn position. On its recorded 35-second segment, precise
RGB-D alignment and a wider fixed image exclusion both worsened localization
error; neither is enabled. The retained keyframe tracker had 0.23 m final error
on that diagnostic segment, versus 2.12 m and 0.45 m for those alternatives.

Transit turn following now retains its turn phase through up to 15 cm estimated
position drift, with 4 cm intermediate-waypoint arrival tolerance for planned
portal travel. This avoids repeatedly undoing a turn to chase small localization
changes. Manipulation docking defaults remain 1.5 cm / 4 cm turn-drift tolerance.
This is a navigation follower setting, not a relaxed task-success criterion.
The physical component test `crossing_route_28_turn_fix` completed the same
observed route in 36.35 wall seconds / 28.84 simulated seconds, with no recorded
external contacts. Its video is `crossing_route_28_turn_fix/crossing.mp4`.
This component starts from a recorded post-opening state. Full run 29,
`closed_two_room_29_success_graph`, crossed without recorded external contact but
failed `undock_endpoint_outside_transit`: its endpoint was within the follower's
tolerance but in a neighboring raster cell outside the narrow portal goal mask.
This is a failed run, not full-task success.

For portal crossing (`erode_goals=False`), completion now uses the planned
geometric exit pose and follower tolerance. Generic undocking retains its
free-region arrival test. The regression for the neighboring-cell endpoint passes;
31 targeted undocking, portal and pipeline tests passed. Full run 30 starts from
the original closed-door scene with this fix. Physical full-task success remains
unverified until the final semantic scorer passes.

Run 30 has now passed full two-room exploration (618.9 seconds including stage
overhead; 175.66 simulated seconds) and the first return to the kitchen (42.4
wall seconds, no recorded external contacts). It recorded 1,607 continuous SLAM
poses into the exploration route graph, reused the kitchen/table labels, and
skipped the extra table-search stage. The graph permits swept-checked local
connections up to 1.5 m between historical visits, so the newly opened leaf does
not force reuse of an obstructed old approach segment. Manipulation and final
full-task success are still pending.

Saved-state return diagnostics built from sparse one-second console poses failed
at floor support near the initial viewpoint. They do not contain the full
continuous SLAM history used by the live pipeline. No floor-support removal or
station-arrival relaxation from those diagnostics is enabled.

Run 30 subsequently passed first-object docking and pickup, but failed its first
carry with `blocked_motion` during a turn. Private physics diagnostics showed
base/bowl contact with the unopened leaf and about 29 cm localization error at
contact. This is a full-task failure despite the earlier successful stages.

The frozen-wall SE(2) registration experiment uses current calibrated nav depth
and previously observed upper-wall cells. On the recorded carry it reduced mean
position error from 20.0 to 7.9 cm (about 3.3 cm near contact), averaging 83 ms per
registration. Simulator poses were used only to measure those errors.
`carry30_corrected_localization` then reached the dining destination in physical
simulation: 146.0 wall / 61.88 simulated seconds, no recorded door contact, and
eight gripper–bowl contacts at the final state. It is a saved-grasp component test.

Full run 31 started from closed doors and integrated that correction into the
continuous tracking session at 2 Hz after the first carry handoff. SIFT remains
initialized once, raw keyframe tracking continues, and the frozen map is not
modified. Thirty-one tracker/route/runtime/pipeline tests plus six localization
and tracker-handoff tests passed. Run 31 nevertheless failed its first carry:
the visual pose drifted, and weak wall matches did not recover it reliably.

The subsequent fix requires sufficient wall-match support and ignores
unobservable directions in wall registration. `PlanarVisualMotion` also bounds
visual increments using issued base commands: a stationary visual observation
still means zero motion, even when forward motion was commanded. It never
reads true base position. Wall registration corrects the resulting estimate
against the frozen observed map; no new mapping or model initialization occurs.
The saved-grasp component `carry31_planar_localization` passed in 106.33 wall /
53.28 simulated seconds, without recorded door contact and with eight final
gripper–bowl contacts. Eleven motion, localization, tracking and base tests passed.

Full run 32 (`closed_two_room_32_planar_localization`) is validating the change
from the original closed-door state. Exploration passed in 643.8 wall seconds,
the first kitchen return in 45.3 seconds, and first-bowl docking in 55.8 seconds.
These stage results do not yet establish full-task success. In particular,
`status: scanning` in the exploration log denotes ongoing execution, not failure;
run 29's terminal failure must not be attributed to a newer active run.

