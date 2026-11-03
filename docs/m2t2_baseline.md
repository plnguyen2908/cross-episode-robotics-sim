# M2T2 manipulation baseline

Experimental sensor-only alternative to the arm/gripper VLA. It does not replace
SLAM or base navigation and is not yet a qualified composite executor.

## Architecture and information boundary

Semantic instruction → existing sensor-based docking → onboard RGB-D points →
pretrained M2T2 grasp targets → robot IK / collision-validated smooth motion →
native MolmoSpaces arm/gripper controller. The base remains held during this test.

`M2T2Policy.predict(points, rgb, held_points=None, ee_pose=None)` is the reusable
model interface. Inputs are metric coordinates in a gravity-aligned robot-base
frame. Optional held-object points and tool pose support the upstream placement
inference path; placement execution has not been tested. Its outputs are candidate
poses, not an executable arm policy or a guarantee of physical success.

The smoke test restores private saved simulator state only to initialize the
benchmark. Execution receives fresh navigation/wrist RGB-D, calibrated robot
geometry, robot proprioception, the saved **estimated online SLAM** pose and the
semantic map's observed bowl location. No simulator object pose, object mesh,
segmentation truth or ground-truth collision map feeds inference/planning.
Object lift and contacts are inspected only by the private final scorer.

Only the two onboard views are rendered: navigation RGB/depth on top, wrist
RGB/depth below, at normal simulated speed. Physical camera-mast collisions remain.

## Upstream and environment

- Repository: https://github.com/NVlabs/M2T2
- Tested source revision: `be2e5f6feae2f961615ab0f7b643f0f98b0c3a50`.
- Weights: `wentao-yuan/m2t2`, `m2t2.pth`, revision
  `a20325dcd19cd1b838274974d0ebe35bfe383796`.
- Generic point-cloud pick/place checkpoint, not the language/RLBench checkpoint.
- `.venv-m2t2` reads the existing local PyTorch 2.7.1/cu126 installation and adds
  dependencies separately. It does not modify π0.5's environment.
- PointNet++ is compiled from upstream using a locally installed CUDA 12.6
  compiler and the existing CUDA library headers. No model-source patch.
- M2T2 CUDA allocator budget is 18% of one GPU; inference exits before that
  test's cuRobo stage. Sharing GPUs with training can still reduce throughput.

```bash
cd /nobackup/le/cross-episode-sensors
bash scripts/setup_m2t2.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/test_m2t2_pick.sh \
  runs/docking_distance_smooth1/close_forward
```

The shell prints a unique output directory. It stores raw point inputs,
provenance, scored candidates, chosen target if any, physical report and video.

## Frame and execution conventions

Upstream M2T2 predicts Panda gripper-root poses with a 0.1034 m displacement
along its approach axis to the contact center. The adapter converts this to our
calibrated grasp-site convention and maps M2T2's closing x-axis to our tool y-axis.
The conversion's orthogonality, contact-center displacement and approach direction
have a regression test. Different physical gripper fingers remain a transfer gap.

Candidate selection uses proximity to the observed semantic target, robot reach,
confidence, and pose diversity. The revised smoke test checks at most 40 diverse
candidates, not all hundreds of predictions. A failed search therefore does not establish that no
possible grasp exists. No oracle grasp is used as a fallback.

The revised default uses a 2 cm approach, a small upward/backward rim-clearance
motion, and a retreat/lift. It tests reachable wrist orientations while preserving
the M2T2 grasp pose. It checks all phases before moving, rather than discovering
an unreachable lift after grasping. No automatic tuck/untuck or redocking is added.

`ObservedGraspPlanner` uses calibrated robot IK and cuRobo robot kinematics/self
collision geometry. The default trajectory is a quintic joint curve with 1 rad/s
velocity and 2 rad/s² acceleration bounds, sampled at 50 Hz. Every executed target
and additional samples at <=0.001 rad summed-joint intervals are checked against
self collision and observed cuboids using the established 5 mm clearance margin.
Blocked curves are rejected, not executed. Upstream cuRobo trajectory optimization
is an optional fallback and is **disabled by default** because it stalled in the
initial shared-GPU test. Thus this is M2T2 + sensor-derived geometric execution,
not a claim that the default motions are cuRobo-optimized trajectories.

Adjacent occupied cells are merged into cuboids without filling holes or deleting
occupied cells. A tested KD-tree broad phase accelerates sphere/box checks; it
matches the brute-force signed-distance predicate. Validation runs in small
batches to fit beside π0.5 training.

The target contact allowance is refined from its connected observed depth surface
above the measured support plane, rather than a fixed-radius region around a
possibly biased map centroid. Countertop and neighboring object points remain.
Commands are interpolated at 50 Hz; settling uses measured joint error/velocity
and finger stability. The final one-second hold is a grasp verification window.

## Limitations

The observed target contact volume is excluded from the planning world to permit
intentional grasp contact. Real simulator collisions remain enabled. This is a
coarse contact allowance, not a link-specific contact planner. Held-object collision
attachment and place execution are not implemented in the smoke-test executor.
The benchmark scores sustained lift (>8 cm) and pad contact over one
simulated second. Its private contact checker resolves nested collision bodies
to the free-object root; checking only the root body missed the bowl meshes. It tests a bowl pick, not transport, placement or set-table.
A successful inference call must never be reported as a successful grasp.


## Initial failed smoke-test results — November 2, 2026

Tests ran while the user's eight-GPU π0.5 training continued. Wall times are
therefore not standalone performance measurements.

| Docked input | Inference incl. loading | Target-adjacent proposals | Diverse candidates checked | Successful picks |
|---|---:|---:|---:|---:|
| Close forward | 4.70 s | 245 | 16 | 0 |
| Farther forward | 3.55 s | 180 | 16 | 0 |

**0/2 completed pipeline trials succeeded.** Both stopped during collision-checked
trajectory selection, before physical action (0 simulated seconds). Some targets
failed at approach and some at the grasp pose. The saved videos contain only the
initial observation, not a grasp attempt; use the input images rather than treating
these as demonstration videos.

Additional close-view IK diagnostics: initial self clearance 18.6 mm and observed
world clearance 286.6 mm. The original 8 cm approach did not yield IK solutions for
the tested candidates. Reducing it to 2 cm yielded reachable candidates (e.g. 8,
9, 14, 15 in the saved ranking); this does not certify their complete paths.
The short-approach joint-goal trial was stopped after several minutes in cuRobo
trajectory optimization / CUDA event synchronization, before executing motion.
It is **inconclusive**, not an additional completed physical trial. GPU contention
may affect time; we have not established it as the cause of planning rejection.

Artifacts: [summary](../runs/m2t2_pick_v1/summary.json),
[close input](../runs/m2t2_pick_v1/close/observation.png),
[farther input](../runs/m2t2_pick_v1/farther/observation.png),
[close report](../runs/m2t2_pick_v1/close/physics/report.json),
[farther report](../runs/m2t2_pick_v1/farther/physics/report.json).

Next qualification work is reachable candidate selection coupled to docking,
full observed-world path validation, then physical grasp retention. Only after
that should placement and complete set-table be evaluated. Those initial results did not establish a successful physical demonstration.
The repaired pick trials below supersede that initial status, not the original artifacts.


## Repaired pick execution — November 2, 2026

**Both repeated physical tests passed (2/2), in the same scene from two docked
viewpoints. This is a smoke test, not an estimated general success rate.**

| Case | Simulated execution | Bowl lift | Sustained hold |
|---|---:|---:|---|
| Close forward | 8.64 s | 15.19 cm | 50/50 control frames with pad contact |
| Farther forward | 6.84 s | 15.18 cm | 50/50 control frames with pad contact |

Both selected the first candidate in the new reach-aware ordering after validating
all phases. The network predictions and pretrained weights were unchanged.
The repaired code addresses: overly broad candidate deduplication; reach-aware
ordering; the fixed 8 cm approach; a fixed contact exclusion missing the bowl rim;
a direct lift that exceeded reach; expensive per-voxel collision boxes; unbounded
optional optimizer use; GPU validation batch size; and nested-body contact scoring.
The original 5 mm clearance margin, support plane and real physical collisions
are retained. The scorer still requires the same object to remain raised >8 cm
with pad contact on every frame of the final one-second window. The root/child
lookup fix does not substitute lift alone for contact validation.

Each motion uses a bounded quintic joint curve, not an unchecked interpolation.
Approach, grasp, rim withdrawal, and lift curves were checked before execution.
No tuck/untuck or base motion was inserted. Settling holds use measured readiness;
the close-gripper and one-second verification windows are retained for grasp
stability. Video runs at normal simulated speed. Planning/inference wall time
is additional and is not included in the simulated execution durations.

- [Close video](../runs/m2t2_pick_v7/close/rollout.mp4)
- [Farther video](../runs/m2t2_pick_v7/farther/rollout.mp4)
- [Summary](../runs/m2t2_pick_v7/summary.json)
- Reports, chosen M2T2 predictions, compressed observed world and validated joint
  paths are in each case directory.
- Six regression tests pass, including exact voxel coverage, collision broad-phase
  equivalence, motion speed/acceleration limits, target component separation,
  grasp-frame conversion, and private child-body contact ownership.

π0.5 training continued throughout. No training/model configuration was changed.
Placement, cups, held-object collision attachment, fresh-scene generalization and
full composite execution remain unqualified.

## Fresh-scene navigation → pick trial — November 2, 2026

Selected held-out two-room `set_table_000002`, whose room doors were already
open. Reset preserved the recorded integration state and initial travel posture.
The GPU test uses Qwen3-VL-8B on one GPU and perception/navigation on another.
The first room-gated diagnostic was interrupted: an observed bowl was available
but room confirmation kept the robot scanning. The successful rerun used an
object-only semantic goal (`goal_room=''`), without supplying a room pose or route.

Sensor-based docking succeeded in 9.12 simulated seconds, with no recorded
external robot-contact events. No One-Ring movement was required in this case:
the observed target triggered the geometry-based docking stage. Navigation's
terminal label was `docked_reachability_unverified`; this is permission to run
M2T2's full arm-path checks, not a certificate of grasp success. The runner now
accepts verified sensor docking plus an observed target for this planning stage.
It still rejects failed navigation, and M2T2 must validate approach/grasp/lift
before any arm movement.

M2T2 produced 135 target-adjacent proposals; two candidate approaches were rejected
before a complete path was found. Physical pick succeeded in 7.40 seconds, lifting
the bowl 13.35 cm and retaining pad contact during the final one-second hold.
This is one successful local navigation-to-pick trial in a held-out source scene,
not a full set-table success or a cross-room transport test. Carry/place and
held-object collision attachment remain unqualified.

[Combined normal-speed video](../runs/m2t2_pipeline_gpu_v2/navigation_and_pick.mp4)
· [Summary](../runs/m2t2_pipeline_gpu_v2/summary.json).

Run with `CUDA_VISIBLE_DEVICES=0,1 VLM_DEVICE=cuda:1 bash scripts/test_m2t2_set_table.sh`.
The script defaults to GPU inference, never silently falling back to CPU.

## Carry-to-placement continuation — November 2, 2026

The pick executor now saves its actual final integration state for continuation.
Repeated the successful held-out pick and restored that physical held-object state
for a carrying test; no object was moved to the table by reset. Experimental
`ObservedPayload` derives a rigid tool-relative envelope from pregrasp depth and
the executed grasp target. Its 2 cm padding is not proof that unobserved object
surfaces are covered. The base bridge accepts a non-stowed posture only when an
explicit observed swept-motion guard approves each moving command. Normal callers
retain the original stowed-posture requirement. Exploration supports a configured
expected retained posture; this alone never certifies clearance.

The first coarse-grid check rejected payload/counter overlap. A finer 5 cm XY /
2.5 cm Z grid cleared this, but rotation required unobserved floor support. Reused
previous estimated SLAM footprints and conservatively transferred observed floor
cells, with fresh local obstacle sensing. Retracing the observed final approach
allowed backing away with the bowl, but the following loaded turn was rejected
at 2.5 simulated seconds for `unobserved_base_support`. Physical collisions stayed
enabled. This is a failed carry/docking test: no dining table was selected and
**placement was not executed**. The new worker's placement proposal option is
implemented but not physically qualified. Placement execution still needs work.

[Grasp and departure video](../runs/m2t2_dock_place_v1/pick_and_departure.mp4)
· [Summary](../runs/m2t2_dock_place_v1/summary.json).
The first interrupted carry diagnostic spent time stopped on `arm_not_stowed`;
its artifacts are not a successful motion trial. Subsequent v2–v5 reports retain
each failure. 25 existing navigation/M2T2 tests and two payload geometry tests pass.

## Placement fix and physical component test — November 2, 2026

**One isolated placement passed in 5.56 simulated seconds.** Source: held-out
two-room set-table episode 000002, recorded first pre-place state. The private
diagnostic reset restores recorded joint/object positions and commands with zero
velocity. This is not a continuous navigation → pick → carry → place result.

Execution uses freshly rendered onboard RGB-D, calibrated robot geometry and
proprioception. Grounding DINO proposes the table; Qwen3-VL verifies the furniture
region; measured depth provides its surface. M2T2 proposes placement transforms.
No destination coordinates, scene object bindings or oracle placement pose enter
execution. The base remains fixed throughout this component test.

Fixes:
- Use a furniture-specific verification prompt. The earlier vessel-focused prompt
  mislabeled the foreground bowl rather than the table.
- Reconstruct the held vessel from nearby cup/bowl detections in either onboard
  view, excluding the robot mask. Previously the wrist bowl's “cup” label caused
  selection of gripper pixels from the other view.
- Explicitly allow only calibrated stationary base spheres to contact observed
  floor voxels. Arm/payload spheres and non-floor obstacles remain checked.
  This option is off by default and is supplied only by the fixed-base placement
  test. It is not permission for base motion through unknown space.
- Diversify candidate contact positions instead of exhausting near-duplicate
  proposals. Validate approach, lowering and retreat in robot/observed geometry,
  including the observed held-object points before release.

The physical test selected its first candidate, used bounded quintic joint
commands at 50 Hz, measured arm/release settling, then checked the initially held
bowl for sustained table support, no gripper contact and low velocity for one
second. Simulator truth is used only for this private scoring. The bowl is near
the table edge; this trial does not establish robust placement margin or
generalization. Nine focused planner/payload tests pass, including a regression
that the fixed-base floor exception does not exempt a moving sphere or an obstacle.

[Normal-speed placement video](../runs/m2t2_place_fix_v1/place_v5/rollout.mp4)
· [Physical report](../runs/m2t2_place_fix_v1/place_v5/report.json)
· [Final onboard views](../runs/m2t2_place_fix_v1/place_v5/final.png).

Reproduce the isolated test after preparing its private reset:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/nobackup2/le/.cache MUJOCO_GL=egl OMP_NUM_THREADS=4 \
PYTHONPATH=$PWD:/nobackup/le/cross-episode-robotics-sim/external/molmospaces \
.venv-slam/bin/python scripts/test_m2t2_place.py \
  --dock runs/m2t2_place_fix_v1/isolated_start --isolated-initial-state \
  --output runs/m2t2_place_new
```

Carry remains unresolved. A collision-checked wrist-floor-view motion retained
the bowl in 6.90 simulated seconds. The robot self-mask now includes base/mast
geometry and both cameras, eliminating a false self-obstacle. Nevertheless,
the following turn stops after 2.40 seconds because a corner of the swept base
requires unobserved floor support. That guard was not weakened. See
`runs/m2t2_place_fix_v1/carry_mask_fix/report.json`; do not interpret the successful
placement reset as a successful cross-room docking continuation.

### Deeper placement trial

Added `observed_placement.supported_interior`: require the observed vessel's
horizontal radius plus 6 cm of measured tabletop support around a candidate.
The check samples the interior and perimeter; it does not fill unknown holes with
a convex hull. Nine placement/planner tests pass, including edge/hole rejection.
The same fixed dock retained 2,035 proposals with a 17.56 cm support radius, but
the first 40 spatially diversified candidates had no feasible IK solution. No
physical commands were executed (`runs/m2t2_place_deeper_v1/report.json`). This is
not a successful deeper placement or proof that all possible poses are unreachable.

Next integration: One-Ring approaches the visually identified dining table;
observed RGB-D/SLAM docking selects a closer base pose jointly with an interior
placement/reachable arm approach. Carrying requires swept arm/payload clearance
and observed floor support. The previous 5.56-second placement remains the only
qualified placement result; loaded redocking has not yet passed end to end.

### Position and facing correction under test

The local docking test now requests a base heading toward the interior placement
TCP, rather than retaining the original heading. For the saved observation this
is approximately 19.6 cm forward and −41.6 degrees. This is a diagnostic local
candidate, not yet a general loaded-navigation solution. Its actual execution
still uses online SLAM and the observed base/arm/payload swept guard. The first
physical attempt stopped before translation at unobserved floor support
(`runs/m2t2_redock_facing_v1/report.json`); no facing success is claimed.

Placement candidate search now interleaves wrist-orientation groups rather than
spending all 40 attempts on one orientation. The current extra support margin is
3 cm beyond the observed horizontal vessel footprint (the earlier trial used
6 cm). Neither setting accepts the original unsupported edge footprint.

View-preparation experiments retain physical grasps and try to expose floor
before moving the base. Below-table viewing tests use an intermediate lateral
waypoint before lowering. Payload clearance now supports an exact oriented-box
separating-axis test, retaining the padded observed envelope, rather than a larger
bounding sphere. The fixed-support sphere lookup was accelerated with an exact
Chebyshev-distance KD-tree lookup; its tolerance and exemptions are unchanged.
Robot masks can also use measured depth with calibrated robot bounds, so a robot's
projected silhouette does not erase visible background at another depth. These
changes do not supply simulator scene labels or world poses to execution.

The exploratory scripts are `prepare_m2t2_place_floor_view.py` and
`test_m2t2_place_redock.py`. They remain component diagnostics, initialized from
a private pre-place reset. They are not integrated into the full One-Ring carry
pipeline. Interrupted superseded trials are not successes. The recorded 5.56 s
edge placement is still the last qualified placement until a new physical report
and video explicitly say otherwise.

### Physical outcome of the facing/redocking experiments

The oriented-envelope lower-view motion passed in 5.54 simulated seconds
(`m2t2_place_floor_view_v7`, repeated in v8), and the collision-checked return to
the earlier carrying posture passed in 5.48 seconds (`m2t2_place_raised_v1`).
Both required sustained private pad-contact retention. These are viewing/posture
successes, not placement successes.

Retaining the measured floor map allowed a short forward movement, but the
straight redock stopped at 1.40 seconds on an observed obstacle near the base
(`m2t2_redock_facing_v3`). Added `test_m2t2_place_redock_astar.py` to use the
existing position/heading A* planner with the retained arm and observed payload.
It re-anchors measured floor and previously occupied SLAM footprints, transforms
the observed target consistently, and samples facing docks with robot-only reach
checks. The final trial `runs/m2t2_redock_astar_v2/report.json` rejected all 15
sampled docks: nine observed collisions and six unknown-support regions. No
safe loaded route was found. The diagnostic assumes its perception artifact and
input map share the original frame; it is not a general production interface.

**Deeper placement is still unqualified.** Do not report the candidate heading
change as a successful physical turn or a complete fix. The original isolated
reset does not carry the earlier approach map; end-to-end integration must keep
SLAM state and enough local observations through navigation and manipulation.
17 focused sensor, placement, payload and planner tests pass. No changes were
merged to main, and the original edge-placement video remains the last successful
placement artifact. Superseded interrupted trials v5/v6 and the interrupted direct
placement v3 are not successes.

## Successful alternative dock and deeper placement — November 3, 2026

Replaced the narrow, single-target docking trial with observed interior targets
and full-circle facing candidates (`navigation/placement_docks.py`). A shared
multi-goal A* search checks routes to alternative docks; each candidate still
requires observed base/arm/payload clearance and robot-only reach. The intended
placement point is passed to M2T2 as a preference, not an oracle success region.

A physical left/right wrist scan retained the bowl and returned the arm, taking
28.36 simulated seconds. The scan is over the normal 15 s target because it
includes two collision-checked lower-view/return motions and four 1 s retention
checks. Mapping retained all observations through those motions. Table memory is
associated with the initially recognized table plane and footprint using actual
reconstructed points; holes are not filled with a model mesh. It is re-anchored
with estimated SLAM poses for docking and placement.

`runs/m2t2_alternative_docks_v2`: 60 feasible candidates, successful local dock in
2.92 simulated seconds, held-bowl retention passed. Heading error was 0.93 degrees
to the selected *docking target*. The shortest successful candidate moved closer
and selected a different, frontal interior target; it did not execute the older
42-degree turn. The old `advance_requested`/`heading_requested` fields in this
artifact describe the superseded proposal; `route_events` records the executed
candidate. New reports omit those misleading legacy fields.

`runs/m2t2_alternative_place_v3`: physical placement passed in 6.88 simulated
seconds with stable table support, no gripper contact and low velocity for the
final second. The chosen M2T2 contact is about 21.7 cm farther forward than the
old edge placement, comparing in the original estimated frame. M2T2 selected a
nearby contact about 13 degrees off the robot centerline; this is not a claim of
perfect final-contact or table-edge-normal alignment. This is one successful
local scan → dock → place continuation from a private pre-place reset, **not** a
full navigation → pick → cross-room carry result.

A placement-cloud bug omitted the robot mask. Applying it exposed depth samples
up to 13.5 mm outside calibrated robot bounds; self-mask padding is now 20 mm
(previously 10 mm), matching the existing minimum depth-consistency tolerance.
This changes perception filtering, not physical collisions. The original arm
configuration now passes the observed-world check. The depth-aware mask was
accelerated by testing only projected image regions: 1.21 s → 0.29 s per capture
on this case, with identical masks before the separate padding adjustment.
21 focused tests pass; broader qualification remains outstanding.

[Dock and deeper placement — 10 s, normal speed](../runs/m2t2_alternative_place_v3/dock_and_deeper_place.mp4)
· [Placement report](../runs/m2t2_alternative_place_v3/report.json)
· [Final views](../runs/m2t2_alternative_place_v3/final.png)
· [Complete wrist scan](../runs/m2t2_placement_scan_v1/rollout.mp4).

### Far-start approach correction — November 3, 2026

The first 1 m farther-start test stopped at tabletop perception before running
One-Ring. This was a test-ordering failure, not evidence that One-Ring could not
find a table. `scripts/test_m2t2_far_pipeline.sh` now orders the stages as:
One-Ring RGB approach with continuous RGB-D SLAM → fresh table verification and
surface reconstruction → observed-geometry docking → M2T2 placement.
One-Ring completion is only permission to inspect the surface, never evidence
of docking or placement success. Failed perception still prevents manipulation.
The approach retains the configured carrying posture and requires swept checks
including the observed payload; no unconditional tuck is inserted. Its map is
fresh and simulator geometry is not exposed to the policy. The local docking
stage currently rebuilds its map to avoid mixing coordinate frames; persistent
map handover remains a limitation. Initial validation: 18 navigation tests pass;
physical far-start qualification is in progress under `runs/m2t2_far_ring_v1`.

### Far-start recovery debugging — November 3, 2026

The farther trial exposed two perception issues: the broad detector query missed
an obvious dining table, and its detected cup-sized patch represented only a
small part of the held bowl. Table-specific proposals now still require VLM and
depth verification. Already-held object reconstruction uses connected onboard
depth near the measured tool after robot self-filtering, rejects insufficient
or boundary-truncated surfaces, and retains the resulting padded geometry for
swept collision checks. It does not establish grasp success.

One-Ring now stops after repeated guarded rejections; it cannot rotate into
unobserved floor. Fresh verified tabletop geometry can hand over to observed-map
docking. If observations remain insufficient, physical wrist scanning is the
recovery path. The first scan inputs left bowl surface in the obstacle cloud;
these runs are failed diagnostics, not qualified demonstrations. The corrected
scan is `runs/m2t2_far_scan_v4`. Run status must be read from its report, not this
implementation note. Observation capture preserves payload artifacts even when
loaded from a preceding stage. Size-bucket broad-phase collision queries retain
the same exact sphere/box narrow-phase test; brute-force equivalence tests pass.

### Farther-start physical result — November 3, 2026

The 1 m backward private initialization now has a successful physical continuation:

| Stage | Simulated seconds | Result |
|---|---:|---|
| Wrist floor scans and return | 31.00 | Both views completed; held bowl retained |
| One-Ring retry | 1.10 | Eight turn commands blocked; stopped for recovery |
| Observed-map A* dock | 27.42 | Success; held bowl retained; target heading error 1.33° |
| M2T2 placement | 7.18 | Success; released bowl stably supported on table; no planning rejections |

This is **not a successful pure One-Ring approach**. Its guarded turns still
encounter unobserved rear-corner support. The wrist scans supplied enough floor
evidence for a forward A* route, which performed the actual approach. No floor
holes were declared free, and no simulator coordinates were supplied to planning.
The tested chain starts with the private already-held bowl initialization, then
scan → One-Ring → A* → placement; it is not a full pick/carry/set-table episode.
The shell runner includes an initial One-Ring attempt before recovery, but that
entire shell invocation has not independently been qualified end-to-end.

The scan and docking exceed the usual 5–15 s motion target: the scan acquires two
blind-side views and returns while retaining the bowl; docking uses conservative
0.08 m/s motion and stops at intermediate route nodes. These are remaining speed
limitations, not evidence that all motion defaults are fully met. Video is normal
simulation speed, without removed pauses. Total simulated chain time: 66.70 s;
encoded video: 67.20 s including stage-boundary frames.

- [Full sequence](../runs/m2t2_far_place_v1/full_sequence.mp4)
- [One-Ring attempt, A* approach and placement](../runs/m2t2_far_place_v1/approach_and_place.mp4)
- [Final view](../runs/m2t2_far_place_v1/final.png)
- [Machine-readable stage reports](../runs/m2t2_far_place_v1/chain_report.json)

Validation: 59 tests pass across hybrid navigation, observed-map continuation,
shaped docking, observed payload, M2T2 collision checks, and observed placement.
Changes remain on the sensor-integration branch; no merge is implied.
