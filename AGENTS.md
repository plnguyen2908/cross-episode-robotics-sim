# Repository working requirements

## Mandatory atomic motion defaults — user decision, October 18, 2026

Read [the motion requirements](docs/human_like_motion.md#mandatory-defaults)
before changing atomic skills, shared motion components, composite execution,
or demonstration generation. These requirements apply to standalone and
composite executions, including local navigation inside fixture skills.

- Target 5–15 simulated seconds per typical atomic call. Do not pad faster
  skills. Long routes, difficult contact, and explicit task waits may exceed the
  target; record the reason rather than compromising correctness.
- Automatically omit redundant tuck/untuck and redocking only when a swept
  collision check validates the retained posture and subsequent motion.
- Replace fixed idle holds with measured readiness and sustained contact or
  stability checks. Never omit required grasp/release verification.
- Blend compatible free-motion phases; preserve necessary contact boundaries.
- Do not impose a fixed compression-factor ceiling. Calls already under five
  seconds need no further compression. Derive stronger retiming for longer
  calls from the time budget and validate it physically.
- Choose retiming from planned trajectory length, turns/contact geometry, and
  the remaining per-call duration budget; do not apply one fixed multiplier
  blindly to every skill.
- Retime eligible motion and interpolate executed commands at 50 Hz within
  velocity, acceleration, tracking, collision, and grip-retention limits.
- Validate faster execution in physics and back off when it fails. Never
  substitute video acceleration, teleportation, disabled collisions, or relaxed
  success checks for actual improvement.
- Keep these behaviors enabled by default in shared executors, including
  internal re-navigation, retreat, reach, and posture changes. Composites must
  continue to call the atomic interface.
- Report measured per-skill duration, retained holds/posture changes and their
  reasons, and physical success. Enabled flags alone are not proof that a
  demonstration is optimized.

This file is persistent project guidance for future coding sessions. The policy
is mandatory; it does not assert that every existing implementation or saved
video already satisfies it. Track and disclose remaining implementation gaps.

## Robot observation defaults — user decision, October 21, 2026

New robot observations and demonstration videos use exactly two onboard views:
`robot_0/navigation_cam` (base-fixed mast) and `robot_0/wrist_cam`.
Fixed room cameras are diagnostic-only, never planner/VLA observation inputs.
Retain physical camera-mast collision geometry and check clearance; do not hide
mounting collisions by disabling them. Existing saved videos/models are legacy
artifacts and are not automatically converted or physically requalified.

Camera-mount changes require checking the arm's motion envelope, not only the
initial tuck: run saved-trajectory clearance checks and fresh composite physics
tests before calling a batch ready. Thin poles must retain conservative but
shape-appropriate planner collision geometry; coarse AABB spheres can falsely
block otherwise feasible arm motion.

## Demo observation modalities — October 21, 2026 update

Two physical RGB-D cameras produce four synchronized demo panels in a 2×2 grid:
navigation RGB and depth on top; wrist RGB and depth below (fixed 0–8 m visualization).
Keep raw metric depth separate from its color visualization. Never substitute
fixed room cameras for these panels. Preserve camera-mast collisions.

## Randomized composite generation — October 21, 2026

The default variation script must use the randomized pipeline, not the fixed
seven-case coverage suite. Sample the base-house seed/count before task controls;
resolve room-door identities from the accepted graph, never assume three doors.
Preserve before/after furniture circulation checks and final scene validation.
Never silently fall back to the old three-room house when sampling fails. Save
candidate seeds/rejections and distinguish static validity from physical success.
Resume preserves saved configurations and successful episodes; new distributions
require a new batch. See docs/randomized_composites.md for controls and artifacts.


## Sensor-only semantic mapping scope — November 1, 2026

Hydra/DAAAM integration focuses on manipulable objects, initially cups and bowls.
Keep furniture/receptacle recognition separate for navigation or placement.
SLAM/observed occupancy and live depth supply collision geometry: retain all
observed table, leg, chair and wall geometry regardless of semantic labels.
Do not interpret excluded semantic labels as free space. Native Hydra's estimated
place/room regions may support routing when linked to repeated visual room labels
and observed free-space routes. They are estimates, not simulator room truth.
Never use simulator object identities, true poses, or room graphs as perception inputs.

## Sensor SLAM default — November 3, 2026

Use the shared `create_mapper()` factory for sensor execution: CUDA SIFT, CUDA
RGB-D fusion, eight OpenCV CPU threads, no added frame subsampling. Runtime is
`.venv-sift/bin/python` (setup: `scripts/setup_gpu_slam.sh`). Preserve explicit
backend/device/thread overrides; never silently fall back to CPU or DA3. See
`docs/rgbd_slam_optimization.md` for measured quality and remaining validation.

## Door-state invariant execution — November 8, 2026

The closed-door baseline is the open-door pipeline plus the shared door-opening
backend when exploration needs passage to an unexplored room. Do not introduce a
separate closed-door transit, carry, docking, or placement policy. Initial door
configuration belongs only to test reset; onboard observations decide whether
opening is needed. After nested door motion, refresh the sensor packet and SLAM
pose, refresh the affected doorway map, and resume the same exploration engine.
Retain its semantic memory and traveled-route graph. Qualify both initially open
and closed full episodes; a successful opening or crossing alone is insufficient.

## Contact tracking diagnostics — November 8, 2026

User decision: arm/base tracking deviation and progress-monitor thresholds must
not abort an otherwise executable door command. Log these as diagnostics and
continue the command sequence; do not require a 5 mm endpoint match after a
completed command chunk. Preserve collision-aware planning, physical collisions,
grasp/release verification and unchanged private task scoring. Do not interpret
removing tracking aborts as physical success. Standalone continuations must
restore available observed map context, without simulator-derived geometry.

## Frozen-map transit — November 8, 2026

User decision: cross-room transit and carry must execute the graph/A* route on
the frozen observed map without a second live-depth obstacle veto, temporary
live-obstacle inflation, or a live-depth detour rejection. Keep localization,
the planner's frozen-map obstacle/footprint handling, physical collisions and
semantic scoring. This applies to both cached routes and target-route fallback.
Local manipulation/docking perception is a separate skill, not a transit gate.

## Preserve the qualified set-table baseline — November 10, 2026

User decision: use `set_table_workspace_full07` as the working baseline and do
not modify its execution behavior while qualifying the other household tasks.
It passed 33/33 stages and final semantic scoring. The source snapshot and
SHA-256 manifest are in `runs/five_task_sensor_20261110_baseline`.
Fix added fixture/appliance/generic-receptacle skills and non-set-table recipe
integration separately. Preserve the qualified exploration, object approach,
pick, frozen-map carry, free-space placement and set-table dispatch. Do not
hot-edit a running pipeline's implementations. A component pass is not a full
composite pass; report each task's physical result independently.

## Household evaluation continuation — November 10, 2026

User decision: added sensor household recipes attempt every planned call rather
than aborting the whole episode on a failed skill or diagnostic. Preserve failed
outcomes; final task success comes exclusively from the unchanged private
semantic scorer. Do not fabricate observations, successful grasps, IK solutions,
or missing handoff outputs. Keep the live physical state after partial execution.
Remove extra held-target/empty-hand orchestration vetoes in these recipes; required
sensor acquisition and executable planner outputs still constrain individual
motions. Fixture-return docking uses frozen-map base-footprint planning without
a second arm/gripper 3D veto. Preserve physical collisions and native A*/cuRobo
planning. This is evaluation behavior, not weaker datagen acceptance. Preserve
the qualified set-table implementation. `--fail-fast` restores legacy recipe aborts.

## Qwen approach before manipulation — November 11, 2026

For the Qwen pick/place docking and redocking paths, the first dock only moves
near and faces a sensed target. Do not gate that approach on a grasp orientation,
arm IK, supported placement footprint, or full insertion/release trajectory from
the distant view. Preserve observed-map navigation and retained robot/payload
swept collision checks. Capture fresh close-range RGB-D after physical arrival,
then reconstruct and plan manipulation; Qwen may redock on that actual feedback.
Navigation anchors are not release surfaces. Preserve the qualified non-Qwen
set-table path. Report docking and physical manipulation success separately.

## Full-task Qwen dispatch — November 11, 2026

The Qwen branch's full task runner must use QwenSkillInterface over the existing
navigation/exploration handlers, not silently bypass it with the bare M2T2
interface. Pickup and fixture/control calls request fresh local Qwen-ranked
redocking by default; placement owns its loaded docking and interior refinement
inside the skill, avoiding duplicate preliminary motion. Preserve observed map,
payload and live-state handoffs. Log the actual interface and redock decisions.
Keep the prior qualified non-Qwen set-table behavior accessible with
`--legacy-m2t2-interface`; changing the Qwen dispatch is not requalification of
all five tasks. Test each physical recipe separately.
