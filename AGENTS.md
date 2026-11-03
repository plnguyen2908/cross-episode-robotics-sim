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
