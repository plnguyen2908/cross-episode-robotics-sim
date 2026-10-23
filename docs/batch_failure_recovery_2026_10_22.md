# Recovery changes for randomized composites — October 22, 2026

Source batch: `runs/composite_pilot_k8yxo0si`, 21 successful and 14 failed
examples. User requested code fixes without simulation reruns.

Changes:

- Same-room navigation tries checked reverse departures of 15, 25, 35, then
  50 cm when the direct departure fails. Each route still requires swept
  robot/payload clearance. Failed searches restore the previous undock setting.
- Object/support docking includes additional lateral offsets. Each candidate
  is checked for an executable forward route before committing navigation.
- Composite recovery recognizes cuRobo grasp-planning failures and checked
  local door-dock failures, requesting a distinct explicit navigate dock before
  retrying the manipulation. Recovery requires successful checkpoint rollback.
- Carry navigation gets at most one conservative-timing retry after an actual
  drop or failed loaded-tuck grip check. The atomic runner restores physics and
  rebuilds the planner first. Timing settings are restored afterward; unrelated
  geometry/infrastructure failures do not trigger a duplicate slower attempt.
- Missing/unstable release support gets at most one slower placement retry.
  For ordinary table placement, the retry requests a 5 mm release clearance
  instead of 4 cm. The full loaded path must remain collision-clear and measured
  support/stability must still pass. Compartment/basket geometry keeps its own
  placement implementation; no containment criterion is relaxed.

No contacts, collision geometry, success thresholds, or recorded videos were
changed. These are bounded recovery improvements, not proof that all 14 examples
now succeed. Only code-level regression tests were run; the user owns the next
physical batch run. Existing successful examples remain untouched on resume.

Validation: 63 focused regression tests passed (motion backoff, composite
recovery, forward-route checks, door motion, and checkpoint execution). Python
compilation and whitespace checks passed. No rollout or video rerun launched.

## Follow-up: 26 passed, 9 failed

Inspection of `composite_pilot_k8yxo0si` after the user's rerun found three
additional code defects. These fixes have **not** been tested in a simulator
rollout, following the user's request to handle reruns personally.

- **Unwrapped navigation yaw:** local A* converted travel headings to the
  principal angle while the robot yaw could contain multiple revolutions.
  Swept interpolation then tested a long rotation instead of the equivalent
  short turn. Travel and docking headings now unwrap relative to the preceding
  pose. Collision checks remain enabled along the resulting short turn.
- **Carry preflight mismatch:** retaining a loaded arm posture checked the raw
  route but not its forward-facing conversion. It now uses the same shared
  preflight as fixture docking before caching a route or committing the retained
  posture. A rejected conversion falls back through the existing posture logic.
- **Rotated receptacle footprint:** placement sampling used a world-axis bounding
  box, which includes unsupported corners for rotated slabs. Candidate footprints
  now also have to fit inside the slab in its own coordinate frame. This is a
  conservative geometry check; existing execution and support checks still apply.

Validation: 73 focused tests passed across navigation/door preflight, rollback,
recovery, carry posture and rotated support footprints. `git diff --check` passed.
No batch, physics rollout, or video generation was launched. These tests establish
code behavior, not a new batch success rate. The failed grasp, grip-retention and
placement executions still require a user rerun to establish physical success;
these changes do not demonstrate that all nine failures are resolved.

## Follow-up: six remaining failures (29/35 passed)

The following patches target the remaining reports in `composite_pilot_k8yxo0si`.
They are code fixes and recovery changes, not six newly successful rollouts.

| Failed example | Evidence and change |
| --- | --- |
| Set table #7 | Bowl grasp reached bilateral contact but its lift hit the counter. The internal withdrawal then failed and masked that error. Atomic pick now lets the checkpoint runner restore before another physical attempt; object-relative rejected grasp poses survive annotation-library changes. |
| Clear/store #3 | Predicted retreat clearance passed 5 mm, but measured/rebuilt clearance was 4.962 mm. Retreat selection now requests 10 mm planning clearance while preserving the 5 mm acceptance threshold. Navigation planning failures can also request a different dock through the atomic interface. |
| Clear/store #5 | Cabinet insertion missed its TCP target by 16.9 mm, with shelf contacts. A bounded, slower checkpoint retry now covers contact tracking errors; cabinet retries also plan a 15 mm release gap rather than 5 mm. Entry collision checks and post-release support/stability remain mandatory. |
| Clear/store #7 | The error combined posture and grip failures. In fact both pads retained contact and slip was about 2 mm, but joint 7 missed the tuck by 2.19 radians. Static inspection of saved failure state found link5/link7 penetration of 0.88 mm. Default loaded tuck paths now receive the same native-mesh preflight as alternate postures; previously the default was exempt. |
| Groceries #1 | A cuRobo failure inside `navigate` stopped the composite without trying another dock. Bounded navigation recovery now issues an alternate atomic navigate and treats its success as completion, without repeating the original call. |
| Groceries #6 | Loaded placement recovery could return to its current or previously tried dock. Recovery now excludes those poses, and keeps a per-object/support dock history so empty-arm recovery cannot cycle either. |

No collision exclusions, weaker tracking tolerances, relaxed grasp/support
criteria, or video speedups were introduced. No physics rollout was launched;
only saved-state inspection (`mj_forward`, no stepping) and regression tests.
A resumed user batch must still establish whether the physical failures pass.

Validation after these patches: **98 tests passed** across the remaining-batch
regressions, motion backoff, composite recovery, carry posture, checkpoint
execution, grasp alternatives, supported lift recovery, forward routing and door
motion. Compilation and `git diff --check` passed. Physical success remains
unverified until the six failed episodes are resumed.

## Follow-up: three remaining failures (32/35 passed)

- Clear/store #7: loaded-carry candidates now include three elbow/wrist folds,
  not only wrist-yaw changes. All candidates still require joint-limit, cuRobo,
  and native robot/payload swept checks. Retained carry checks up to four
  destination docks before concluding that the posture has no route.
- Groceries #1: the apparent navigation failure followed a basket placement
  whose empty-hand withdrawal had failed. Basket placement no longer reports
  success with a trapped hand. The atomic runner rolls back and tries a slower
  placement at a distinct valid release region (at least 6 cm away), then can
  request a different dock. Neither release support nor withdrawal collision
  checks are bypassed.
- Groceries #6: basket docking now includes approaches 6 and 10 cm closer than
  the first supporting-counter dock. Elevated basket floors otherwise require
  more reach than those generic counter docks provide. These candidates must
  pass the same complete route and collision checks; the parent counter remains
  in the collision world.

Scheduling now defaults to eight independent episode workers, with exclusive
idle-GPU leases. It is no longer limited to one concurrent example per task.
The old task-lane helper remains for compatibility, but generation uses the
shared episode queue. Each episode saves `gpu_assignment.json`; resume still
keeps successful folders and exact saved configurations.

No simulation rollout or generation batch was launched. Static arm-pose checks
are not proof of a feasible trajectory or retained payload. The three examples
still require physical validation on the user's resumed run.

Validation: 40 focused tests passed for scheduling (including eight concurrent
same-task episodes), carry posture, placement recovery, withdrawal and composite
recovery. Shell syntax and whitespace checks passed. Physical success remains
unverified for the three failed examples.

## Follow-up: two grocery failures (33/35 passed)

- Groceries #1 now reaches the second snack-box navigation, after the previously
  blocked placement/withdrawal. Local A* previously required turning at the final
  dock. If that route fails, it now tries an alignment point 25, 40 or 60 cm before
  the dock, followed by a straight forward approach. Both the route to alignment
  and every 25 mm of the final connector must pass the existing floor and exact
  robot/payload clearance checks. This can address a valid dock with no room to
  rotate there; it cannot make a physically blocked dock valid.
- Groceries #6 includes gripper spring-link/rim collisions during descent, not
  only IK errors. Basket release height is now derived from the actual gripper
  collision vertices transformed into the proposed release orientation: the
  lowest hand point must remain 8 mm above the rim. Only the minimum additional
  vertical offset is applied. The solid object settles under gravity; existing
  containment, support, settling and empty-hand withdrawal checks still decide
  success. No object poses or collision settings are modified to force success.

No simulation reruns were launched. These changes require a resumed physical
run before claiming the two examples pass.

Validation: 22 focused tests passed (shared navigation, basket geometry,
compartment withdrawal and batch recovery), including a narrow-dock straight
entry regression and minimum hand/rim clearance calculation. Python compilation
and whitespace checks passed. The recorded success count remains 33/35 until
new physical runs complete.

## Monitored physical follow-up

The next user run reached the second snack-box placement in both grocery
examples but still failed. Saved-state IK checks showed that a 15-degree lean
admits a solution at a counter dock where the upright orientation did not.
The shared placement search now offers 15/30-degree lean candidates for rigid
`snack_box` objects in baskets; all containment, swept-mesh, release, settling
and withdrawal checks remain mandatory. Basket docking also offers lateral
positions with the chassis square to the counter, avoiding an unnecessary
turn toward the target at a tight dock.

Saved failure-state continuations physically passed:

- #6: `runs/basket_tilt_physics_3m0gufiu/result.json` — navigate 4.520 s,
  place/withdraw 2.040 s.
- #1: `runs/basket_one_tilt_physics_i7r62yrl/result.json` — navigate 15.958 s
  (includes rejected retained-posture planning and clearance/tuck recovery),
  place/withdraw 2.268 s.

These are diagnostic continuations, not complete episode passes. The user
explicitly requested monitoring and fixes, authorizing physical reruns for the
two failures. A full resume was launched preserving the other 33 successes.
Regression checks: 17 carry/docking/recovery tests passed; the earlier 12 basket
geometry/withdrawal/recovery checks also passed.

### Full physical rerun results

Both previously failing grocery examples now complete all atomic calls and pass
the task goal in their original saved randomized scenes: #1 completes 20/20
calls and #6 completes 16/16. The other 33 successful examples were preserved.
Final second-snack-box placement/withdrawal takes 2.05 s (#1) and 1.92 s (#6).

All navigation, pick and place calls in these two runs are under 15 simulated
seconds. The two room-door opening calls in #1 remain over target (60.05 s and
59.15 s): direct panel-push approaches failed IK, requiring local repositioning,
handle-grasp fallback and subsequent physical panel pushing. These retained
motions are logged in `run.log`; this is an outstanding duration-optimization
gap, not evidence that every atomic call meets the mandatory timing target.
No collision, containment or grip-retention checks were relaxed.

Artifacts are in `runs/composite_pilot_k8yxo0si/` under
`unpack_and_sort_groceries_000001_sampled/` and
`unpack_and_sort_groceries_000006_sampled/`: `skill_calls.json` contains measured
call timings; `report.json` contains outcome and physical diagnostics;
`composite.mp4` contains the rendered demonstration.

Final monitored resume: **35/35 successful**, exit after 21 min 41 s wall time.
Both new videos finalized and passed ffprobe (H.264, 1280×960):

- [Grocery #1 video](../runs/composite_pilot_k8yxo0si/unpack_and_sort_groceries_000001_sampled/composite.mp4): 272.32 s.
- [Grocery #6 video](../runs/composite_pilot_k8yxo0si/unpack_and_sort_groceries_000006_sampled/composite.mp4): 120.72 s.

The batch has 35 successful episode reports. This validates these saved scenes;
it does not guarantee success for every future randomized house.
