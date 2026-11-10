# Sensor-only door-family tests

## Current qualification status

Both room leaves, cabinet, fridge, and microwave now have complete opening,
release, and withdrawal passes. The final left-leaf regression also passes. These are five
prepositioned **component cases**, not a general success-rate benchmark or a
qualified autonomous public `open_door` skill.

| Case | Qualified run | Simulated time | Result |
|---|---|---:|---|
| Right room leaf | `door_oracle_transfer_right_10` | 68.42 s | Pass, 89.9° |
| Cabinet | `door_oracle_transfer_cabinet_05` | 39.64 s | Pass, 89.5° |
| Fridge | `door_oracle_transfer_fridge_10` | 41.38 s | Pass, 90.0° |
| Microwave | `door_oracle_transfer_microwave_13` | 43.68 s | Pass, 90.0° |
| Left room leaf | `door_oracle_transfer_left_05` | 61.46 s | Pass, 87.7° |

All five sampled contact audits found no non-target contact and no non-gripper
target contact. These are sampled recording states, not every physics substep. Videos run at physical speed. Times exclude
the initial grasp, but include contact preparation, repositioning, opening,
release and withdrawal. They exceed the usual 5–15 s target; contact validation,
floor acquisition for the room doors, and incremental base/arm synchronization
remain duration costs. No claim of complete motion optimization is made.

## Initial batch (historical)

Five independent prepositioned component trials, November 5, 2026. Both room-door leaves are from the same double doorway and were tested independently, not sequentially in one episode. This is a small coverage test, not a benchmark success rate.

Existing M2T2 grasps seed the left leaf, cabinet, fridge and microwave. A fresh right-leaf grasp passed in 8.12 simulated seconds before its opening trial. The opening times below exclude those grasp sequences. Execution uses onboard observations, calibrated robot geometry and estimated base motion. Fixture identities and angles below are private post-execution scoring only. No fixed cameras or simulator target poses enter execution.

| Case | Full opening | Final angle magnitude | Opening trial duration | Video |
|---|---|---:|---:|---|
| Room door: left leaf | Pass | 89.2° | 66.44 s | [Video](../../cross-episode-robotics-sim/runs/navigation_previews/door_family_room_left_01.mp4) |
| Room door: right leaf | Fail | 10.7° | 22.46 s | [Video](../../cross-episode-robotics-sim/runs/navigation_previews/door_family_room_right_01.mp4) |
| Cabinet | Fail | 34.7° | 13.76 s | [Video](../../cross-episode-robotics-sim/runs/navigation_previews/door_family_cabinet_01.mp4) |
| Fridge | Fail | 34.0° | 13.16 s | [Video](../../cross-episode-robotics-sim/runs/navigation_previews/door_family_fridge_01.mp4) |
| Microwave | Fail | 33.7° | 22.28 s | [Video](../../cross-episode-robotics-sim/runs/navigation_previews/door_family_microwave_01.mp4) |

The right leaf failed contact-preserving reposition IK after obtaining floor observations. Cabinet, fridge and microwave exhausted collision-checked push approaches; partial openings are failures. The successful left-leaf trial includes release and withdrawal and exceeds the 5–15 second motion target. Drawer is a separate skill and was not included.

Do not advertise a general door-opening skill from these results. Next work is a dock that also changes base heading for the opposite leaf, and fixture-specific observed contact/regrasp selection for appliance and cabinet doors. The public M2T2 interface and oracle implementations remain unchanged by this test batch.

Machine-readable results: [report](../runs/door_family_report_01.json). Individual run folders contain physical states, logs and private score reports. Videos are rendered afterward at physical speed.

The successful left-leaf sampled audit found no non-target contact and no non-gripper target contact in 589 push-phase recorded states. This is a sampled audit, not an assertion about every physics substep.

## Oracle-procedure transfer follow-up (in progress)

Components with recorded passes: left room leaf, cabinet, fridge, and microwave (4/5).
The left-leaf latest-code regression is still being repaired; do not read this
as a 4/5 pass rate on the current code.
The original table above records the first batch, not the latest status.
Right leaf remains unqualified until a complete physical opening,
release, and withdrawal passes.


The oracle's procedures are being adapted at the geometric-input boundary,
not called with simulator hinge/handle/base truth. Added checked empty-arm
repositioning (the released hand need not remain fixed in world space),
signed push roll/tilt candidates, base-following retained-handle motion,
and an optional exact closed-finger static collision narrow phase.

`door_oracle_transfer_cabinet_04` passes full opening/release/withdrawal:
89.5 degrees, 39.62 simulated seconds, with a retained handle grasp and base
following the observed hinge arc. It exceeds the motion-duration target and
starts from a verified grasp. This supersedes the cabinet failure above only
for this tested configuration. Right-leaf attempt 03 reached 77.8 degrees but
failed wrist/panel clearance; it remains a failure. Fridge and microwave are
still under test. No general public door handler has been qualified.

The numeric IK seed is clipped to solver bounds without changing measured
state or robot limits. A short aligned Cartesian contact approach can be tried
when a single joint-space path fails; intermediate swept collision checks are
retained. These checks and the exact-box broad phase have regression coverage.

Fridge follow-up `door_oracle_transfer_fridge_09` also passes full opening,
release and withdrawal: 90.0 degrees in 41.40 simulated seconds. It uses a
retained grasp, coordinated base translation, and an additional final opening
increment; the private scorer is unchanged. Its 363 sampled push/base-follow
states show only gripper contact with the target, no non-target contact, and
no non-gripper target contact. [Video](../../cross-episode-robotics-sim/runs/navigation_previews/fridge_sensor_open_success_09.mp4).

The base-yaw extension transforms both tool position and orientation into the
rotating base frame. Its translation must subtract the displacement already
caused by base rotation; a regression test covers this coupling. Rotated base
support and full robot clearance remain checked. This extension is experimental
until its physical run passes; it is not needed by the qualified cabinet/fridge
translation-only configurations.

### Base-turn allocation correction

Right-leaf attempt 06 exhausted arm IK late in the push. Microwave attempt 09
never executed its intended coordinated turn: the full-yaw candidate required
unobserved floor cells, and the arm-only fallback eventually exhausted reach.
The contact planner now preflights full, three-quarter, half, quarter, and zero
base contributions to the requested tool yaw. Each candidate compensates its
translation for base rotation and retains observed-floor, IK, and swept robot
collision checks. No candidate changes the requested tool goal or private scorer.
Runs `door_oracle_transfer_right_07` and `door_oracle_transfer_microwave_10`
exercise this correction. Both signs of base yaw have regression coverage.
These runs are pending, not declared successes.

### Latest physical regression results

- Cabinet `door_oracle_transfer_cabinet_05`: pass, 89.5 degrees, 39.64 s.
- Fridge `door_oracle_transfer_fridge_10`: pass, 90.0 degrees, 41.38 s.
- Microwave `door_oracle_transfer_microwave_13`: pass, 90.0 degrees, 43.68 s;
  [video](../../cross-episode-robotics-sim/runs/navigation_previews/microwave_sensor_open_success_13.mp4).
- All three include release/withdrawal; sampled private contact audits report
  no non-target and no non-gripper target contact.
- Right attempt 09 reached the 90-degree hard stop but continued commanding
  motion, triggering tracking failure before withdrawal. It remains a failed
  run. Attempt 10 removes the extra final five-degree increment.
- Left attempt 02 failed a tracking guard after a different contact approach
  was selected. The old passing left video is not evidence that this new
  configuration passed. Attempts 03/04 investigate the regression, with 04
  using calibrated closed-finger boxes for static collision narrow phase.

Observed floor support now persists through estimated base transforms; unknown
ground remains rejected. Optional adaptive panel-push yaw checks reduced wrist
turn increments if the nominal path is infeasible. This applies to a released
panel push, not a retained handle grasp. The focused suite has 41 passing tests.

Left attempt 03 reproduces attempt 02: the alternative push approach fails its
first collision sweep, and the forward reposition fallback lags its commanded
7.5 cm halfway point by 15.6 mm. The 15 mm tracking guard remains unchanged.
Attempt 04 uses calibrated closed-finger static collision geometry to avoid
rejecting a feasible contact approach due to covering spheres. No collision
geometry or simulation contact is disabled.

### Reposition timing correction

Contact-preserving fallback options now try the shortest displacement first.
Longer motions use a 50 Hz quintic profile with duration derived from translation
and yaw length, peak velocity, and acceleration. The previously qualified
12 cm / 2 s repark is preserved; a 15 cm repark takes at least 2.5 s instead
of blindly reusing two seconds. Tracking tolerances and physical scoring remain
unchanged. A regression checks this duration/rate relationship.

### Final left-leaf regression

`door_oracle_transfer_left_05` passes at 87.7 degrees in 61.46 simulated seconds,
including release and withdrawal. It uses calibrated finger boxes and adaptive
push orientation, with the original near-limit scorer unchanged. The target is
near its physical limit, not claimed to be exactly 90 degrees.
The focused regression suite passes 42 tests. All five runs are summarized in
[the machine-readable component report](../runs/door_family_qualified_components.json).

Remaining scope: these experiments start at verified M2T2 grasps. They do not
qualify arbitrary fixture instances, autonomous handle acquisition, closing, or
the public M2T2 `open_door` dispatch. No oracle execution state was introduced.
The oracle interface remains unchanged.

The final left audit covers 539 sampled push states, with only the right gripper
follower contacting the target (maximum sampled normal force 10.91 N), and no
non-target or non-gripper target contact. Final onboard frames were inspected.

## Successful videos

- [Left room leaf](../../cross-episode-robotics-sim/runs/navigation_previews/left_room_door_sensor_open_success_05.mp4)
- [Right room leaf](../../cross-episode-robotics-sim/runs/navigation_previews/right_room_door_sensor_open_success_10.mp4)
- [Cabinet](../../cross-episode-robotics-sim/runs/navigation_previews/cabinet_sensor_open_success_05.mp4)
- [Fridge](../../cross-episode-robotics-sim/runs/navigation_previews/fridge_sensor_open_success_10.mp4)
- [Microwave](../../cross-episode-robotics-sim/runs/navigation_previews/microwave_sensor_open_success_13.mp4)

Wall-clock profiling and optimization: [measured results](sensor_door_wall_time.md).
