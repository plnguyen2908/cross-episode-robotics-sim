# Motion shaping toward human-like demonstrations

Latest implementation and measured exceptions: [atomic duration defaults](atomic_duration_defaults.md).

The objective is to make the robot's executed motion continuous, purposeful,
and reasonably paced. This describes trajectory construction, not the separate
recorded-action replay procedure. Smoothness alone does not demonstrate a match
to human motion; that requires comparison with human demonstrations.

## Mandatory defaults

**User decision, October 18, 2026:** these are required defaults for every atomic
skill, standalone demonstration, and composite execution—not optional experiment
flags. Apply them in shared motion components, including internal fixture
navigation, re-navigation, retreats, and posture transitions.

1. **Duration target:** aim for 5–15 simulated seconds per typical atomic call.
   Do not slow a faster skill to reach five seconds. Long navigation, difficult
   contact, and explicitly requested heating/wait durations are exceptions.
   Record why an exception exceeds the target.
2. **Automatic posture removal:** retain the current arm configuration instead
   of tucking and untucking when the entire subsequent swept motion is
   collision-clear. Keep necessary clearance motions; avoid unconditional
   tuck at skill boundaries.
3. **Automatic pause removal:** replace fixed idle holds with state-dependent
   readiness checks. Preserve sustained grasp contact, release/support checks,
   and explicit task waiting. Do not delete stationary frames from a video.
4. **Continuity:** blend compatible free-space motion segments rather than
   stopping at every waypoint. Keep required contact and feedback boundaries.
5. **Retiming and interpolation:** resample eligible executed commands at 50 Hz
   after shortening movement time within actuator velocity/acceleration,
   tracking, collision, and grasp-retention limits. More interpolation samples
   alone do not increase speed. Use phase-specific timing, not one multiplier
   for every skill or a forced duration that overrides physics. There is no
   fixed compression-factor ceiling; calls already under five seconds need
   no further compression. Allocate the available duration across trajectory
   segments after accounting for their individual motion limits.
6. **Physical validation and backoff:** execute and verify the faster trajectory.
   If it fails, use a slower validated profile and report the exception.
   No teleportation, collision disabling, weakened success criteria, or
   accelerated video playback.
7. **Evidence:** report per-skill simulated duration and time spent in motion,
   holds, and posture changes, with reasons for retained pauses/tucks. Verify
   the behavior in both atomic and representative composite runs.

### Implementation status

The shared defaults now use adaptive operational grasp holds, checked posture
retention, route shortcutting and length-aware timing. The full atomic suite
passed, but not every call meets 15 seconds: wide cabinet opening/closing
remain over target. The latest stronger retiming brought the tested 11.3 m
cross-room carry to 14.8 seconds with physical success. See the
[measured timing table](atomic_duration_defaults.md#measured-per-skill-durations).
Existing recordings remain unchanged; resume skips prior successful videos.

## Procedure

**Atomic skill → optimize internal components → meaningful waypoints → smooth
paths → timing and interpolation → physical execution.**

1. **Optimize the internal components first.** Audit navigation and re-navigation,
   tuck/untuck, reach, contact, release, and settling separately. Remove redundant
   turns/posture cycles, skip already-satisfied docks, and replace fixed idle
   holds with measured readiness. Preserve forward travel, necessary clearance,
   sustained grasp evidence, and explicit task waits. Verify this version before
   attributing further improvements to smoothing.
2. **Define meaningful waypoints for the optimized procedure.** A pick has
   approach, grasp, lift, and withdrawal phases. Door opening follows the hinge
   arc; drawer opening follows its slide axis. Navigation supplies a
   collision-free route. A waypoint can include position, orientation, and a
   required contact condition.
3. **Generate smooth paths between waypoints.** Free-space arm motion uses
   collision-checked planning. Contact motion follows fixture geometry with
   local IK. Interpolation fills the path between targets, with collision
   checking of the interpolated sweep. It must not round a contact path away
   from a handle, hinge arc, or insertion axis.
4. **Choose timing and interpolate commands.** Accelerate through clear space
   and slow near contact. Retiming changes how fast a path is traversed;
   interpolation supplies commands at the control rate. More samples alone
   do not make a movement faster. Respect actuator limits, endpoint tracking,
   grip retention, and required task waiting time.
5. **Connect compatible phases.** Blend approach/lift/withdrawal or route
   segments where feasible, avoiding an unnecessary full stop at every
   waypoint. Preserve grasp confirmation, release, collision, and contact
   feedback boundaries. Combining skills does not remove their preconditions.
6. **Execute and record physical interaction.** Objects and fixtures must move
   through actual robot contact. Record observations and executed commands
   with synchronized timestamps. Exporting the intended VLA action convention
   and fixed-rate observations is a separate training-data requirement.

## What exists and what remains

- Existing skills already define task-specific paths and contact phases.
- The [waypoint pilot](smooth_waypoint_comparison_2026_10_15.md) tested joined
  navigation segments and smooth joint interpolation for two lift phases.
  It improved measured continuity without materially shortening the demo.
- The [live timing layer](smooth_action.md) shares interpolation and timing
  logic across skill executors. Component-only and fast smoothed execution each
  passed the 11 representative demo families, covering all 16 public atomic
  skills, with the documented corrected cross-room retry.
- Internal navigation, posture transitions, gripper preparation, and readiness
  checks are optimized in shared components. Joining motion phases still depends
  on each skill's geometry and contact requirements. These demonstrations do not
  guarantee success or shorter motion for every asset and layout.
- **Coherent variation is planned:** choose grasp/approach direction, compatible
  control points, and speed per attempt, and retain that choice throughout the
  motion. Independent per-frame noise is not the intended approach. Broad
  randomization and intentional mistakes/recoveries are not established here.

## Separate acceleration experiment

[Recorded-action replay](action_replay.md) compresses commands after a successful
execution, then reruns physics and checks diagnostics. The 76.3 → 38.0 second
result belongs to that experiment. It is not evidence that original online
skill execution is automatically twice as fast, nor that motion is human-like.


## Component-only versus smoothed validation

[Shared component audit and live validation](atomic_component_optimization.md)
tracks the full atomic interface. Optimized components and fast smoothing are
opt-in. Use `--optimize-motion-components --no-smooth-action` to test the component policy without
interpolation; `--smooth-action --smooth-action-profile fast` explicitly selects
the subsequent trajectory timing layer.
This separation supports comparisons without confusing shorter procedures with
smoother playback. Neither mode changes explicit `wait` durations.


## Timing results and individual skill clips

The [comparison table](../runs/atomic_component_comparison/README.md) reports
**full demo durations**, summed over their public skill calls. The
[individual video gallery](../runs/atomic_components_smooth_review/README.md)
contains **one atomic skill call per clip**. These are different scopes: a demo
can contain navigation, manipulation, and an explicit wait. Timings are simulated
execution seconds, not generation wall time, and clips preserve normal playback.

After component optimization, smoothing/retiming shortened 10 of the 11 tested
demo families. The toaster-lever demo was the exception:

| Public call | Component-only | Smoothed |
| --- | ---: | ---: |
| `navigate` | 8.700 s | 8.700 s |
| `push_lever` | 15.658 s | 16.228 s |
| `wait` | 1.000 s | 1.000 s |
| **Full demo** | **25.358 s** | **25.928 s** |

Thus the [push-lever clip](../runs/atomic_components_smooth_review/push_lever.mp4)
is approximately **16 seconds**, not 26 seconds. The full demo grew by 0.570 s
(2.2%); navigation and explicit waiting were unchanged. Smoothing is not a
promise that every motion becomes shorter, nor proof of human-like behavior.

Source calls: component-only
`runs/atomic_components_verified/toaster-lever/skill_calls.json`; smoothed
`runs/atomic_interface_suite_faz0zvlv/toaster-lever/skill_calls.json`.
See the [component audit](atomic_component_optimization.md) for qualification
scope, retry provenance, and the limits of this engineering comparison.


## Latest optimized videos

Local generated artifacts, played at normal speed. Each clip below contains one
public atomic skill call, rather than the entire demo.

| Atomic skill | Optimized video |
| --- | --- |
| `close_door` | [Watch close_door](../runs/atomic_components_smooth_review/close_door.mp4) |
| `close_drawer` | [Watch close_drawer](../runs/atomic_components_smooth_review/close_drawer.mp4) |
| `close_lid` | [Watch close_lid](../runs/atomic_components_smooth_review/close_lid.mp4) |
| `insert` | [Watch insert](../runs/atomic_components_smooth_review/insert.mp4) |
| `navigate` | [Watch navigate](../runs/atomic_components_smooth_review/navigate.mp4) |
| `open_door` | [Watch open_door](../runs/atomic_components_smooth_review/open_door.mp4) |
| `open_drawer` | [Watch open_drawer](../runs/atomic_components_smooth_review/open_drawer.mp4) |
| `open_lid` | [Watch open_lid](../runs/atomic_components_smooth_review/open_lid.mp4) |
| `pick` | [Watch pick](../runs/atomic_components_smooth_review/pick.mp4) |
| `place` | [Watch place](../runs/atomic_components_smooth_review/place.mp4) |
| `push_button` | [Watch push_button](../runs/atomic_components_smooth_review/push_button.mp4) |
| `push_lever` | [Watch push_lever](../runs/atomic_components_smooth_review/push_lever.mp4) |
| `rotate_knob` | [Watch rotate_knob](../runs/atomic_components_smooth_review/rotate_knob.mp4) |
| `slide_oven_rack` | [Watch slide_oven_rack](../runs/atomic_components_smooth_review/slide_oven_rack.mp4) |
| `turn_lever` | [Watch turn_lever](../runs/atomic_components_smooth_review/turn_lever.mp4) |
| `wait` | [Watch wait](../runs/atomic_components_smooth_review/wait.mp4) |

The latest wide-cabinet rerun uses component optimization followed by live
smoothing/interpolation. These additional clips show that fixture specifically:

| Wide cabinet | Optimized video |
| --- | --- |
| `open_door` | [Watch wide-cabinet open_door](../runs/atomic_components_wide_cabinet_review/wide_cabinet_open_door.mp4) |
| `close_door` | [Watch wide-cabinet close_door](../runs/atomic_components_wide_cabinet_review/wide_cabinet_close_door.mp4) |

These `runs/` links require this workspace; generated videos are not published
to GitHub or Google Drive by this documentation update.
