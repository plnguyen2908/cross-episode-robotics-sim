# Cup pickup recovery and IK workspace

November 9, 2026. Sensor worktree; no oracle/datagen changes.

## Failure and correction

The restored run-57 execution path completed both bowl transfers in
`runs/five_task_sensor_20261109/set_table_run57_regression`, then stopped at the
first cup. Its generic error said no collision-checked reachable grasp, but
the detailed trace had **zero target proposals**, no rejected IK candidates,
and zero simulated motion. This was not evidence of an IK or collision failure.

M2T2 randomly subsamples 16,384 points from the observed local cloud. On this
155,833-point input, samples 0–2 proposed 328 grasps elsewhere and none within
the existing 16 cm target filter. Sample 3 proposed 81 near the cup. The
previous successful run's observation was slightly different and produced
178 target proposals in its usual three passes.
Re-running that old input with the fixed worker reproduced exactly 178 target
proposals out of 792 total, with three passes and no fallback sampling.

`propose_pick_grasps()` keeps the original three passes whenever they produce
target proposals. Only an empty result retries further seeds, up to eight total
passes. It reuses the original RGB-D and persistent model. Target contact
filtering, collision checking, joint limits and grasp verification remain intact.
Logs now distinguish no model proposals, rejected target contacts and failed
motion plans.

Measured on the failed input: four passes, 81 target proposals, approximately
1.05 s model/sampling wall time. This is a bounded recovery, not a guarantee that
every observation will yield a grasp.

## Physical evidence

`scripts/test_recorded_pick_retry.py` restores the recorded failed pickup state
as a private test reset, builds a fresh local map from onboard RGB-D, and calls
the existing public pickup backend. No scene mesh or true object pose is passed
to execution. The local map differs from the original episode's accumulated map;
this is component evidence, not a full regression.

- Output: `runs/five_task_sensor_20261109/cup_retry_01`.
- Physical success: true; cup lifted 0.243 m, retained through final hold.
- Simulation: 17.64 s; 50 Hz control, ordinary 1× video.
- Eight candidate plans rejected before accepting a checked approach/grasp/lift.
- Measured settling: 0.10 s approach, 0.16 s grasp, 0.10 s lift.
- Finger stability: 0.32 s; final hold verification: 1 s.
- Above the typical 15 s target because the checked approach/lift paths take
  6.84/7.82 s. No additional tuck or idle padding was introduced.
- Video: [cup pickup](../runs/five_task_sensor_20261109/cup_retry_01/cup_pick_success.mp4).

The full initial-state regression `set_table_run57_retry` passed both bowl
pickups but stalled on the second carry. It was stopped at a repeated
`waypoint_route_changed` loop, before reaching the cup test. The follower treated
a required corner 1.31 cm away as reached (old 3 cm threshold), then rejected
the shortcut and repeatedly planned the identical route. The corner must remain
a target until within a grid-scaled tolerance, capped at 5 mm; the frozen map,
collision inflation and actual command checks are unchanged.

`carry_corner_02` restores that recorded physical state and observed map and
uses the same route follower and base actuator. It progressed 0.53 m with one
plan in 7.22 simulated seconds / 13.26 s wall time. Private post-run scoring
confirmed gripper contact and retained bowl height (change -0.97 mm).
This verifies corner progress, not a completed carry. The first component attempt
failed because its harness omitted the usual carrying motion-monitor callback;
the corrected harness uses the same monitor as the pipeline.

The new full initial-state regression is `set_table_run57_retry02`. Its result
must be checked separately; the other four household tasks remain unqualified.
The proposal/skill-interface/workspace tests passed 50 checks; navigation,
fixed-map and route-memory tests passed 36 checks after the corner fix.

## Reachability visualization

`control/reachability.py::ReachabilityEvaluator` accepts calibrated robot
kinematics, points in the base frame and a gripper orientation. It calls the
installed cuRobo IK solver with 32 seeds, 5 mm position tolerance, 0.05 rad
orientation tolerance, joint-limit and robot self-collision checks.

This sweep uses cuRobo's IK solver directly. The existing pickup fast path uses
calibrated local numerical IK plus cuRobo robot collision geometry, so their
finite-seed convergence can differ. The atlas is a reachability prior, not a
prediction that the existing local solver will always find the same solution.

`scripts/visualize_robot_reachability.py` samples a 3D volume and a finer
horizontal slice at the observed cup's height. It compares fixed top-down and
side grasps. For each point it tests grasp, 6 cm pregrasp, and a 6 cm up/back
lift endpoint. This is a tool-contact workspace, not the robot's 11-dimensional
action vector, nor a guaranteed object-volume envelope.
The sampled volume is the forward manipulation region, not the entire space
behind or around the mobile robot.

Colors distinguish grasp-only IK from all three endpoints. Darker interior
regions are farther from failed samples on the finite grid. Unsolved samples
are not proof of mathematical impossibility. A single fixed orientation per
grasp family does not cover all wrist rotations or object shapes.

The map excludes furniture, other objects and payload geometry. It cannot
certify connected approach/lift paths. Use it to rank candidate docks, then
validate the actual learned grasp and full motion against current sensed
geometry. Moving closer blindly can put the arm in an inner unreachable region
or collide with a counter. Do not replace physical validation with a workspace
membership test.

The atlas is diagnostic and reusable. An **opt-in workspace docking ranker** is
now integrated into hybrid navigation; it remains disabled in the ongoing
baseline regression. See the physical component result below.

Measured plots: [top view](../runs/five_task_sensor_20261109/reachability_01/reachability_top_view.png)
and [3D workspace](../runs/five_task_sensor_20261109/reachability_01/reachability_3d.png).
At the observed cup centroid (0.603, -0.010, 0.992 m in the base frame), the
tested fixed top-down orientation failed endpoint IK; the tested fixed side
orientation passed all three endpoints. The physically successful M2T2 grasp
was also nearly horizontal (86.9 degrees away from straight down), although
its yaw and exact contact point differ from the atlas's representative side
orientation. The full 2,856-point volume plus height slices took 226 s offline.
Re-rendering with cached volumes is a different timing measurement.

For docking, rank collision-feasible base candidates by interior workspace
margin for the intended grasp orientation, then validate the learned grasp,
approach and lift in the observed scene. A robot-only atlas can be computed
once per calibration; live obstacle checks cannot be replaced by it.

## Integrated workspace-aware docking

`navigation/workspace_docking.py::WorkspaceDockRanker` now ranks the frontal
base candidates used by `ShapedDocking`. It transforms the observed target into
each candidate base frame, tests representative top-down and side orientations,
and evaluates grasp, pregrasp and lift endpoint IK at the center and six axis
perturbations. The sampled interior radius takes priority over sampled success
fraction and base travel. It ranks candidates within the existing distance band
and calibrated approach-IK checks. The observed-map swept base route check selects an
executable candidate; M2T2 then proposes actual grasps from a fresh observation.

Enable with `SENSOR_WORKSPACE_DOCKING=1` when launching
`scripts/run_sensor_household.py`. The calibrated ranker is cached in the live
episode runtime. It uses robot calibration, onboard observations and estimated
pose, not simulator target coordinates. The original oracle/datagen interface
is unchanged. This integration currently covers the hybrid frontal pickup
docking path, not all fixture or placement docking paths.

Physical component command (use a fresh output directory):

```bash
PYTHONPATH=.:external/DAAAM/src CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl \
  .venv-sift/bin/python scripts/test_recorded_pick_retry.py \
  --pick runs/five_task_sensor_20261109/set_table_run57_regression/transfer_03_cup/pick \
  --output runs/five_task_sensor_20261109/workspace_dock_01 --workspace-dock
```

This command also requires the existing M2T2 model-service endpoint in
`SENSOR_MODEL_ENDPOINTS`. The saved simulation state is used only for private
test reset. Execution builds its local map and target evidence from RGB-D.

| Component | Result | Simulated time | Wall time |
| --- | --- | ---: | ---: |
| Workspace-ranked dock | Passed; final target distance 0.550 m | 3.52 s | 18.89 s |
| Cup pick after docking | Passed; 24.0 cm lift; zero rejected grasp plans | 15.10 s | See pickup log |

Ranking accounted for 11.45 s of docking wall time. The pickup kept measured
settling, sustained finger stability and hold verification; its approach,
grasp and lift paths retain the existing motion checks. The 15.10 s duration
slightly exceeds the typical target, rather than dropping those verifications.

The previous isolated pickup passed in 17.64 simulated seconds with eight
rejected plans. Including the new dock, this test took 18.62 simulated seconds,
so it is **not evidence of a total speed improvement**. Several candidate
margins saturated the tested 24 cm perturbation radius; this run does not
establish a larger before/after margin or robustness across scenes. These are
sampled endpoint margins, not continuous collision-clearance certificates.

Video: [workspace docking and cup pickup](../runs/five_task_sensor_20261109/workspace_dock_01/workspace_dock_pick.mp4).
The four onboard panels are rendered after physical execution at normal speed.
The workspace/hybrid/docking/reachability regression suite passed 68 tests.

## Frozen-map search recovery regression

`set_table_workspace_full01` finished the first bowl transfer, then stopped
looking for the second bowl after 659.7 s total wall time. Its second search
reported `exploration_required` after 17.04 simulated seconds / 142.76 s wall.
The camera showed the sink/cups initially, followed by a turn toward the doorway.
There was no confirmed bowl and no docking attempt in this stage.

The hybrid navigator unconditionally rejected `search_recovery` on a frozen
map, including routes entirely on observed floor. Recovery now uses remembered
object sightings and the requested station's observed furniture/viewpoints to
propose alternative viewing positions. A* checks each route on the unchanged
map. On arrival, the robot faces the remembered station/object; only fresh
visual confirmation can authorize object docking. These hints do not certify
that an object still occupies its remembered location.

A close dock can also be outside the conservative circular transit footprint.
In that case recovery invokes the existing observed shape-checked departure
before retrying A*, instead of declaring the whole room unexplored. New-space
exploration remains required when there is genuinely no observed route.
Frozen transit keeps its existing no-live-depth-veto policy, localization,
planned collision clearance and physical collisions.

The hybrid/frozen-map/exploration/workspace suite passed 59 tests, including
station viewpoint recovery without fresh target promotion, unchanged map hashes,
unreachable-viewpoint handling and the shared departure handoff.
The first recovery rerun (`set_table_workspace_full02`) exposed a separate
workspace-docking regression before reaching the second-bowl search: removing
the baseline distance and approach checks admitted a farther bowl dock whose
M2T2 approach plans were rejected. The run was interrupted during rejected
grasp planning. Those checks are restored; workspace margin now ranks only
candidates inside the previously validated execution envelope. The sampled
representative grasp joints no longer replace the backend's pregrasp solution.
`bowl_workspace_recovery01` passed from the recorded failed bowl state:
2.62 simulated seconds / 13.04 wall seconds to redock, then 7.60 simulated
seconds to grasp and lift, zero rejected grasp plans, 27.8 cm physical lift.
The final estimated target distance was 0.650 m. This is a local-map component
test, not an identical replay of the accumulated episode map.
[Corrected bowl redock and pick video](../runs/five_task_sensor_20261109/bowl_workspace_recovery01/recovery.mp4).
`set_table_workspace_full03` then tested the initial-state pipeline.

That full run passed both bowl pickups (13.0 s and 13.3 s wall), with the second
search passing directly in 44.3 s; the new search fallback was not exercised.
It then exposed a cached carry route with millimetre-scale points at an inflated
map-cell corner. Re-selecting the same route caused repeated replanning. The run
was interrupted there rather than marked successful.

The fixed-map follower now repairs a failed cached route using A*, retains the
current cell center when connecting the continuous pose, and proactively checks
15 mm tracking margin around cached segments. Routes lacking that margin use
cell-centered A* before entering the corner. This changes the route, not map
occupancy, collision limits or task scoring.

`carry_corner_repair03` and `04` started at the already marginal stopped pose;
they failed after localization drift into the non-transit cell (`04` additionally
used a conservative column extrusion of the recorded 2D occupancy for departure).
`carry_corner_repair05` reset 16 s into the recorded carry, before the corner,
and exercised proactive repair successfully: one plan, 1.219 m progress,
13.52 simulated seconds / 18.92 s wall. Private diagnostics show the held bowl
moved 1.215 m with height change under 0.1 mm. The other bowl stayed on the table.
[Corner repair video](../runs/five_task_sensor_20261109/carry_corner_repair05/corner_passed.mp4).
`set_table_workspace_full04` subsequently failed on an unhandled placement
turn-drift replanning request. The handler now returns to route planning.
`set_table_workspace_full05` delivered both bowls and picked/carried the first
cup, then stopped at fresh dining-table perception (933.58 s total wall time).
Neither run achieved full semantic success.

### Empty-tabletop pointing correction

Replaying the saved first-cup table-observation frame reproduced the failure:
generic Molmo pointing for "dining table" landed on a previously placed bowl.
The smallest containing FastSAM mask was that bowl (884 pixels); semantic
verification rejected it. The wrist proposal had no usable mask.

Placement now uses `PointMaskDetector(free_support=True)`. It asks Molmo for
an empty, unobstructed spot on the receptacle's horizontal support surface,
away from edges, existing objects and the robot. The resulting detection keeps
the receptacle category for verification and depth reconstruction. Object-search
prompts remain unchanged. This point selects a support mask; M2T2 still proposes
the actual placement and the existing geometric/motion checks still apply.

On the same saved frame, the new prompt selected bare tabletop in both views.
Both masks passed semantic verification and yielded 779 navigation-view and
212 wrist-view support points. This is a sensor-perception replay, not a new
physical placement or full-episode success. Eight pointing/mask tests passed.
Reports: `runs/five_task_sensor_20261109/cup_table_diagnosis/report.json` (before)
and `runs/five_task_sensor_20261109/cup_table_free_spot/report.json` (after).

### Free-space targets guide docking and placement (November 10)

The RGB pointing seed now survives mask selection. Placement projects it using
onboard depth and calibration, retaining it only when it agrees with the measured
support surface. `input.npz:preferred_contacts` carries these base-frame hints to
the docking stage, which transforms them into the estimated SLAM frame.

Before docking, `free_approach_targets` excludes observed contents within the
held object's support radius and height. `placement_docks` tries reachable docks
facing the preferred empty patch first, retaining alternative free patches.
The selected `placement_target_base` guides M2T2 placement candidate ordering;
fresh depth, support and arm/payload path checks remain active at release.
This is shared by the dining-table and generic-receptacle placement stages.
It does not change cross-room A* destinations or require a fixed table ID.

Saved cup-scene evidence: 991 support points, 670 supported candidate centers,
629 unoccupied centers after excluding 41 overlapping observed contents.
Two projected RGB free-space hints were retained. These are candidate counts,
not independently qualified placement positions. Details:
`runs/five_task_sensor_20261109/cup_free_spot_physics03/free_spot_selection.json`.

Physical qualification remains incomplete. The prompt-only full06 run placed
the first bowl successfully, then stopped at second-bowl departure with
`undock_route_unverified`, before reaching the cup. Recorded-state cup retries
passed perception but could not validate a dock from a newly rebuilt local map;
the original in-memory full exploration map was unavailable. Sparse self-filter
and collision-volume supplements did not resolve this and were not retained in
the production docking path. No successful cup placement or full episode is
claimed for the free-space-guided docking change. The 51 targeted perception,
placement, map and route tests pass.

### Frozen-map carry departure correction (November 10)

The full06 second-bowl carry stopped at zero simulated time: the initial
`ObservedUndock(PayloadDocking(...))` still compared the attached bowl with 3D
upper-cabinet map points at z=1.375 m. Recovery instantiated that same legacy
backend. This contradicted the chosen frozen-map base-navigation contract.

Both paths now use `make_carry_departure(..., known_map_only=True)`. It plans
the calibrated base footprint against all occupied frozen-map columns, with
no fresh-depth or held-object 3D veto. It uses the shared navigation speed and
turn settings. Physical collisions, observed-map route planning and localization
remain; this is not an arm-clearance certificate. Non-frozen carrying exploration
retains its separate legacy behavior.

`frozen_carry_repair02` restored the actual failed carry state and the saved
observed 2D map/episode route graph. The robot backed away 0.30 m, followed the
route to the dining-table area and retained the bowl (private physical scoring).
The map hash stayed unchanged. Total: **47.37 s wall, 43.50 s simulated**;
departure completed around 2.04 simulated seconds. The total exceeds the typical
5–15 s atomic target because this is a roughly 5.7 m cross-room carry with turns.
This qualifies this carry component, not placement or full-task success.
26 departure/map regression tests passed.

[Carry video](../runs/five_task_sensor_20261109/frozen_carry_repair02/carry_passed.mp4)
and [physical report](../runs/five_task_sensor_20261109/frozen_carry_repair02/report.json).
Full integration rerun: **`set_table_workspace_full07` passed**, from the initial
scene through final semantic scoring: 33/33 stages, two bowls and two cups on the
dining table, no wrong table items and empty hand. Total wall time was 1560.23 s
(26m00s), excluding deferred video rendering. This is one development episode,
not a test-set success rate.

| Transfer | Carry wall seconds | Carry simulated seconds |
|---|---:|---:|
| Bowl 1 | 34.43 | 21.32 |
| Bowl 2 | 40.38 | 21.02 |
| Cup 1 | 50.00 | 23.42 |
| Cup 2 | 55.19 | 21.42 |

All four carries reached the known destination. Free-space-guided docking and
placement also passed for all four transfers, superseding the incomplete
physical qualification above. Remaining runtime gaps include grasp-candidate
rejections and the first cup's 109.9 s placement stage. Cross-room carries still
exceed the typical 5–15 simulated-second target; no blanket speed-qualification
claim is made. Frozen-map base-route, localization and progress checks remain;
the removed gate is the extra 3D held-payload clearance veto, not physics or
final task scoring.

[Full semantic report](../runs/five_task_sensor_20261109/set_table_workspace_full07/report.json)
and [timestamped log](../runs/five_task_sensor_20261109_set_table_workspace_full07.log).
