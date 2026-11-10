# Opening skills: shared wall-time optimization

November 6, 2026. This supplements [the earlier measurements](sensor_door_wall_time.md).
The objective here is lower **wall time**, not faster video playback.

## Scope

The five hinged tests start from saved, verified M2T2 grasps. They exercise opening,
base-assisted contact, release and withdrawal. They do not qualify autonomous
navigation or handle acquisition. The drawer test includes a prepositioned M2T2
grasp/open/release/regrasp/close/release cycle. Results are one physical trial per
fixture, not a cross-layout success-rate estimate.

## Shared changes

- Try bounded local Jacobian IK first, retaining numerical IK when it cannot meet
  the existing tolerances. Every accepted path still passes swept collision checks.
- Stop searching additional grasp-IK seeds once a feasible complete path exists.
  The qualified drawer grasp retains its original numerical selection behavior.
- Compile exact sphere/rotated-box collision tests with Numba (float64, no fastmath).
  Cache exact static-world queries; invalidate when geometry, margin or calibrated
  support geometry changes. Moving panels are still checked separately.
- Cache exact per-joint rotation matrices during finite-difference IK. Angles are
  not rounded; tests compare full forward kinematics bit-for-bit with the original.
- Upload collision geometry to cuRobo only when its fallback planner is needed.
- Stop room-door floor-view acquisition when observed support covers the intended
  repark. Acquire a targeted view when coverage is missing; retain the final swept
  support/collision validation.
- Stop at the requested observed door angle, accounting for the initial probe,
  instead of commanding an additional full opening beyond the physical stop.
- Keep the drawer cycle in one EpisodeRuntime: one simulator, camera interface and
  planner. The recorded model is shared rather than repeatedly serialized.

An aggressive multi-seed local-IK experiment changed the left-door posture and
failed tracking. It is **not enabled**. The final rerun uses the conservative seed
fallback plus exact arithmetic caching.

## Recording and execution

Camera images required for perception are still rendered during execution.
Diagnostic video encoding occurs afterwards. No video is encoded between open,
release, reposition and close phases. Recording small state arrays is distinct
from encoding camera video. These standalone qualification scripts save diagnostic
artifacts; they are not the production skill-call artifact contract.

Execution uses onboard observations, estimated motion and calibrated robot
geometry. Simulator truth is confined to test reset and private physical scoring.
Collisions, grasp/release checks and 50 Hz commands remain enabled.

## Reproduce

```bash
.venv-sift/bin/python scripts/benchmark_sensor_door_wall.py \
  --output runs/door_wall_check --gpus 0 1 2 3 4
```

The output separates setup/execution, deferred rendering and total wall time.
Tests are unprofiled. Parallel workers share CPU resources, so these are observed
batch timings rather than isolated microbenchmarks.

## Remaining motion-duration exceptions

These contact-rich sensor executions still exceed the 5–15 simulated-second target.
They retain probing, supported base repositioning, contact motion and verified
release/withdrawal. Wall-time improvements do not establish human-like motion or
resolve every retained posture transition. No simulated-time reduction is claimed.

## Final measured results

Final batch: `runs/door_algorithm_08`; drawer: `runs/drawer_algorithm_04`.
All six passed their existing physical checks. 58 focused regression tests passed.

| Component | Setup + execution | Deferred video | Total | Simulated | Result |
|---|---:|---:|---:|---:|---|
| left | 221.49s | 34.12s | 255.61s | 67.64s | Pass |
| right | 255.78s | 35.11s | 290.88s | 69.12s | Pass |
| cabinet | 99.93s | 21.91s | 121.83s | 42.18s | Pass |
| fridge | 92.07s | 20.90s | 112.97s | 41.50s | Pass |
| microwave | 106.84s | 25.79s | 132.62s | 46.46s | Pass |

Drawer full cycle: **188.08s setup/execution**, including grasp, opening, release, regrasp, closing and final release. It is not a single opening call. Video encoding ran after the cycle.

Against the earlier unprofiled setup/execution measurements, left/right room doors fell by 27%/39%, fridge by 51%, and microwave by 42%. Cabinet is 99.93s now; its older 147.96s measurement was profiled, so it is not an apples-to-apples percentage comparison.

Room doors still exceed three minutes. Observed floor acquisition, collision-checked arm repositioning and repeated contact solves remain the main work; these runs do not prove every opening is below three minutes. Earlier profiler totals are nested and should not be added. The exact joint-rotation cache preserved kinematics but did not produce a large end-to-end gain.

Private sampled-state contact audits found no unrelated-object or non-gripper contact during recorded push phases in all five hinged tests. These audits do not cover every physics substep.

### Videos

- [left opening](../runs/door_algorithm_08/left/video.mp4)
- [right opening](../runs/door_algorithm_08/right/video.mp4)
- [cabinet opening](../runs/door_algorithm_08/cabinet/video.mp4)
- [fridge opening](../runs/door_algorithm_08/fridge/video.mp4)
- [microwave opening](../runs/door_algorithm_08/microwave/video.mp4)
- [Drawer open–close cycle](../runs/drawer_algorithm_04/video.mp4)

## Follow-up: right door and complete cycles

Work in `runs/fixture_cycle_work` separates opening-only from full cycles.
Latest verified unprofiled setup/execution timings:

| Test | Previous | Current | Physical result |
|---|---:|---:|---|
| Right room door, open/release/withdraw | 255.78s | 185.34s | Pass |
| Drawer, grasp/open/release/regrasp/close/release | 188.08s | 93.40s | Pass |

Final recordings: `right_final` and `drawer_final`. Video encoding is deferred
and excluded from these execution times. The right-door sampled contact audit
reports no unrelated-object or non-gripper contacts.

The right-door profile (instrumented, therefore not the table's runtime) spent
about 101s preparing the push and 154s in coordinated contact segments. These
include nested planning/collision costs. Concrete costs included 36s scanning
all collision boxes, 30s in moving-leaf correspondence and 23s in NumPy sorts.

New shared optimizations:

- A conservative bounding-volume hierarchy prunes static collision candidates.
  Every retained leaf uses the same float64 sphere/rotated-box predicate. Tests
  compare collision decisions across rotated worlds and different world sizes.
- A rigid-panel spatial index is reused by transforming the query points.
  The exact panel geometry and collision margins remain unchanged.
- Moving-leaf correspondence reuses one reference index, queries static distance
  only for possible moving matches, and avoids querying points already excluded
  by the existing radial constraint.
- Exact row deduplication uses packed bytes instead of structured float sorting.
- Drawer contact follows local IK after grasp selection. Four-centimeter planned
  segments reduce repeated setup; swept collision coverage and 50 Hz resistance
  monitoring remain. The segment does not execute blindly to its endpoint.

Cabinet and microwave now have passing open–close cycles (`cabinet_13`,
`microwave_13`). Opening-only passes are not counted as successful cycles. The old closing loop
replayed an absolute opening arc after base movement; it now closes incrementally
from the observed panel and current estimated base frame. The regrasp/initial-closing failures required correcting robot self-masking and
tracking the observed solid door thickness. No collision checks were disabled.

### Closing correctness fixes

- Close from the current observed panel/base frame, not an old absolute opening arc.
- Preserve the attached observation baseline during release and reapproach.
- Anchor the handle contact cloud to measured grasp translation, retaining sensed
  panel rotation rather than accumulating hinge-center translation error.
- Use calibrated finger geometry at the current aperture for narrow-phase checks.
- Include the finger-link visuals in the robot self-mask. This reads only robot
  joint proprioception and robot calibration; simulator segmentation was used
  privately to diagnose the offending pixels, never supplied to the policy.
  Hardware deployment must supply corresponding linkage estimates/measurements.
- Confirm broad parallel layers across differently oriented RGB-D observations.
  The leaf includes the observed thickness between those faces. Its points remain
  in moving-panel collision checks; they are not deleted from collision geometry.

A full cycle retains release, withdrawal, reapproach, regrasp, closing, final
release and withdrawal. Both appliance cycles reached their physical closing
stops in the private scorer. All tests here remain prepositioned fixture
components, not whole-house autonomous open/close qualification.

### Repeat the cycle suite

```bash
.venv-sift/bin/python scripts/benchmark_sensor_fixture_cycles.py \
  --output runs/fixture_cycles_check --gpus 0 1 2 3
```

Use `--skip-video-render` for execution-only timing. The driver checks physical
report success, not merely process exit code. It does not retry failed runs.

### Retiming experiment — not enabled

An attempt to remove the two-second contact floor passed right-door opening,
but failed complete appliance-cycle qualification. Additional sampled-arm timing
checks also made singular arm paths very slow. That experiment was backed out;
the default retains the previously qualified contact timing. No reported final
result uses the rejected faster profile. The retained improvements are planning,
collision-query and perception/runtime optimizations. This remains a documented
simulated-duration exception under the mandatory motion requirements.

## Accepted final results for the requested four cases

| Component | Scope | Setup + execution | Physical result | Video |
|---|---|---:|---|---|
| Right room door | Open, release, withdraw | **182.86s** (was 255.78s) | Pass | [Watch](../runs/fixture_cycle_work/right_sensor_final/video.mp4) |
| Drawer | Grasp, open, release, regrasp, close, release | **83.59s** (was 188.08s) | Pass | [Watch](../runs/fixture_cycle_work/drawer_sensor_final/video.mp4) |
| Cabinet | Open, release, regrasp, close, release, withdraw | **163.33s** | Pass | [Watch](../runs/fixture_cycle_work/cabinet_13/video.mp4) |
| Microwave | Open, release, regrasp, close, release, withdraw | **194.74s** | Pass | [Watch](../runs/fixture_cycle_work/microwave_13/video.mp4) |

All timings are unprofiled single-run wall time, including process/model startup,
excluding deferred video encoding. Cabinet and microwave previously reported
opening-only times; those cannot be used as full-cycle baselines. Their earlier
closing implementation failed, so no full-cycle speedup percentage is claimed.
The right door is about 29% faster; the drawer cycle is about 56% faster.

Private sampled-state push audits for the right door, cabinet and microwave
found no unrelated-object or non-gripper contacts. These are sampled audits,
not proof of every physics substep or generalization across layouts.

Final regression check: **51 tests passed**, including sensor calibration/masking, collision hierarchy equivalence, rigid-panel queries, drawer motion and contact planning.

## Same-code left/right comparison — November 6

Fresh runs in `runs/door_side_comparison` both passed (separate GPUs, shared CPUs; single trials).

| Side | Setup + execution wall time | Deferred rendering | Simulated duration |
|---|---:|---:|---:|
| Left | 206.18s | 38.80s | 67.62s |
| Right | 188.60s | 38.70s | 69.14s |

The right side is not slower on current code. New nested wall timers in
`test_sensor_hinged_cycle.py` report inclusive and exclusive time. Right-side
exclusive costs: coordinated base/contact work 94.4s; collision-world construction
33.4s; push preparation 19.2s; observations 5.9s; arm physics 11.0s; arm planning
and settling 7.5s. Remaining time includes startup and untimed processing.
Use exclusive values to avoid double counting.

The right routine performs 19 contact push segments with repeated planning,
tracking and collision validation, rather than one continuous precomputed push.
The approximately 69-second video excludes computation latency. Video rendering
is deferred until after execution.

An orientation-continuity trial (`runs/door_orientation_continuity`) passed but
took 189.36s versus 188.60s baseline. It avoided only one rejected candidate,
showing no meaningful speedup. That policy experiment was reverted; timing
instrumentation remains. These tests start from saved fixture/grasp setups, not
full autonomous object-finding qualification.

## Shared oracle contact primitive — November 6

The earlier statement that the oracle uses one uninterrupted push was incorrect:
`fixtures/door_push.py` also advances in small angle increments. Its useful
execution pattern is **one endpoint IK solution → combined base/arm sweep check
→ interpolated physical motion**, rather than a fresh IK solve at every spatial
sample. We extracted that pattern and the contact-pose math into
`control/coordinated_contact.py`. Both the oracle and sensor implementations now
call shared code for panel contact placement, signed hinge progression, and
combined arm/base interpolation. The original oracle collision/feedback backend
remains intact.

Sensor execution uses `ContactRetreatPlanner.plan_oracle_segment`: endpoint IK
uses robot calibration; a dense 2 mm sweep checks observed collision geometry,
known floor support and Cartesian tracking. Execution uses the shared cubic ramp
at 50 Hz, with RGB-D base tracking. No simulator fixture pose, hinge state, or
collision truth enters this execution path. Reset and final scoring are private
harness operations. The sensor approach selection, hinge estimation and tracking
remain sensor-specific; this is not a direct call into the oracle's MuJoCo context.

For qualified room-door pushes, duration follows displacement and arm/base
velocity/acceleration limits instead of the old two-second minimum. Repositioning
and appliance cycles retain their previously qualified timing. No video speedup,
fixture actuation, collision disabling, or success-threshold change is used.
The sub-15-second atomic target remains unmet: observed-floor scanning, safe
contact preparation and the large opening sweep are retained exceptions.

First paired physical trial (`runs/shared_oracle_door_01`):

| Side | Previous execution wall | Shared execution wall | Simulated duration before → after | Result |
|---|---:|---:|---:|---|
| Left | 206.18s | 164.02s | 67.62 → 49.84s | Open/release/withdraw passed |
| Right | 188.60s | 144.13s | 69.14 → 49.14s | Open/release/withdraw passed |

Wall times include setup and exclude deferred rendering (28.18s left, 26.89s
right). Trials share CPUs but use separate GPUs; these are individual measurements,
not statistical performance estimates. Both reach approximately 90 degrees.
Private sampled contact audits found no unrelated-object or non-gripper contacts;
left contact was on the gripper follower, not exclusively fingertip pads.

- [Left video](../runs/shared_oracle_door_01/left/video.mp4)
- [Right video](../runs/shared_oracle_door_01/right/video.mp4)

These are **pregrasped component tests**, including release/reposition/panel push
and withdrawal. They do not establish autonomous handle acquisition or full
composite generalization. Two-sided repeat testing is recorded below.

Room-door benchmark cases select the shared primitive by default. Appliance
cases keep their qualified profile. To reproduce or compare:

```bash
.venv-sift/bin/python scripts/benchmark_sensor_door_wall.py \
  --output runs/shared_door_check --gpus 0 1 --cases left right
# Add --no-shared-oracle-contact for the prior dense sensor contact baseline.
```

The standalone harness accepts `--shared-oracle-contact`; the combined fixture
suite selects it for the right room-door case. This does not silently replace the
production M2T2 interface's unsupported fixture operations with a reset harness.

Repeat on final shared-code version (`runs/shared_oracle_door_02`): left **158.67s
wall / 49.96s simulated**, right **139.73s wall / 49.06s simulated**. Both passed;
rendering added 29.05s and 28.34s respectively. Across the two trials: 4/4
pregrasped open/release/withdraw successes. This establishes repeatability on
these two saved setups, not a population success rate. Sampled audits of the
repeat again found no unrelated-object or non-gripper contacts. Left gripper
follower contact peaked at 20.14 N; right pad contact peaked at 23.26 N.

- [Latest left video](../runs/shared_oracle_door_02/left/video.mp4)
- [Latest right video](../runs/shared_oracle_door_02/right/video.mp4)

Regression: 53 tests passed across shared contact motion, contact retreat,
oracle door pushing, direct door pushing and sensor hinge geometry. Tests retain
unknown-floor rejection, obstacle rejection, unreachable endpoint rejection,
signed left/right stop handling and cubic velocity/acceleration bounds. The
original oracle's interpolation formulas are preserved by the extraction;
additional sensor timing limits apply only to the qualified sensor room pushes.

## Further computation-only optimization — November 6

Two additional changes preserve collision coverage and the motion policy:

1. `sensor_drawer_geometry.unique_voxels` packs integer voxel coordinates for
   scalar sorting. It returns exactly NumPy's lexicographically ordered cells and
   first-occurrence indices, falling back for coordinate spans that could overflow.
   On 463,265 saved observed points, median deduplication time over five calls was
   375 ms → 41 ms. No occupied cell is removed beyond duplicate cells.
2. `compiled_collision.point_box_minimum` computes the same float64 signed
   point-to-oriented-box minimum without temporary full-cloud arrays. It uses
   Numba with fast-math disabled. A 12,000-point microbenchmark measured about
   1.43 ms → 0.063 ms; randomized rotated-box tests match within 1e-12.
   This still checks every panel point, with the same padding and rejection margin.

These microbenchmarks are not whole-skill speedups. Physical paired runs are
recorded in `runs/shared_oracle_door_03` (deduplication only) and
`runs/shared_oracle_door_04` (both changes). The motion-planning policy, simulation rate, visual tracking rate, physical
checks and deferred-rendering policy remain unchanged. This is not a claim of
bitwise-identical trajectories across fresh perception/physics runs. New regression suite: 42 tests passed, covering deduplication,
compiled collision, hinge/base clearance, shared contact, retreat and hinge fits.

Measured final paired results (`shared_oracle_door_04`):

| Side | Previous wall (`02`) | Deduplication only (`03`) | Both optimizations (`04`) | Final simulated time | Result |
|---|---:|---:|---:|---:|---|
| Left | 158.67s | 150.25s | **133.08s** | 49.92s | Pass |
| Right | 139.73s | 131.28s | **129.90s** | 49.08s | Pass |

Execution includes startup but excludes deferred rendering (30.33s left, 29.39s
right). Both final runs reached approximately 90 degrees and released/withdrew.
Private sampled contact audits found no unrelated-object or non-gripper contacts.
These remain single paired measurements on two saved verified-grasp setups.

Collision-world exclusive time fell from 35.8 to 23.4s left and 34.4 to 25.4s
right. Contact-work exclusive time fell from 56.9 to 43.2s left, but only 46.5 to
45.6s right: the compiled point/box optimization benefits the exact gripper-box
path more than the right side's sphere-based path. Do not extrapolate its
microbenchmark speedup to the entire skill. Preparation and sensing remain
material costs; the physical duration has not been shortened in this round.

Latest videos: [left](../runs/shared_oracle_door_04/left/video.mp4),
[right](../runs/shared_oracle_door_04/right/video.mp4).

## November 7: live exploration integration

The live opener reused the shared endpoint/swept-contact implementation, but
`--auto-probe-direction` also selected a feedback-only base-motion executor.
Thus the earlier ~130-second pregrasped tests did **not** qualify the live timing
profile. Direction estimation and motion execution are now separate options.

`cross_episode_sim/control/door_open_command.py` is the shared configuration for
`LiveDoorOpener` and `scripts/replay_observed_door_contact.py`. The `fast` profile
restores planned base increments and adds bounded visual residual correction
against the previous command. It retains the 8 mm tool-displacement progress
bound, 50 Hz execution, observed support/swept collision checks, handle tracking,
and physical release/withdrawal checks. Endpoint feedback remains active; this
is not an unmonitored replay of the older nominal commands.

The shared live configuration now defaults to `fast`. `feedback` remains an
explicit comparison option, not a silent alternative selected by auto-direction.

| Saved-grasp component | Profile | Observed angle | Simulated seconds | Base-contact wall seconds (inclusive) | Physical result |
|---|---|---:|---:|---:|---|
| Left, `closed_room_feedback_contact_57` | feedback | 88.64° | 98.90 | 223.29 | Pass |
| Same left grasp, `closed_room_fast_contact_56` | fast | 88.96° | 62.98 | 145.60 | Pass |
| Right, `closed_room_fast_contact_right_58` | fast | 88.88° | 75.66 | 120.64 | Pass |

The matched left comparison reduces measured coordinated-contact wall time by
34.8% and simulated duration by 36.3%. Right execution measured 141.23 wall
seconds across the opening script, excluding navigation/grasp acquisition and
video rendering. Parallel processes shared CPU resources. These saved-grasp
component tests supply a recorded observed support map, which may contain later
observations; they are not fresh end-to-end exploration qualifications. Both
retain collision/contact checks and exceed the 5–15 simulated-second target due
to incremental sensed contact and supported base repositioning. 27 focused
regression tests passed. No diagnostic videos were encoded during execution.

Fresh integrated qualification: `runs/set_table_pipeline_development/closed_two_room_04_fast`.
This run uses the same shared command builder, automatic direction estimation,
and fast profile from a fresh closed-door scene. Its final result must be checked
separately; component passes do not imply exploration or set-table success.
