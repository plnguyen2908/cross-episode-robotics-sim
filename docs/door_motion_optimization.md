# Door motion optimization — 2026-10-14

**Update:** base assistance is now integrated into the shared interface by default.
See [shared integration and validation](#shared-door-and-drawer-integration);
the earlier opt-in notes below describe the original pilot.

The requested behavior is: open the door initially by its handle, release,
then **push the exposed panel with the gripper**. Do not pinch or pull the edge.
The initial edge-grasp prototype was a misunderstanding and has been removed.

## Removing repeated folding

The wide cabinet previously used six 15-degree handle strokes. Each stroke
released, withdrew, folded the arm, navigated, and reached for the handle again.
The physical pulls themselves lasted about two simulated seconds apiece.

The controller now predicts how far a retained grasp can follow the hinge,
checks that arc against native geometry, and continues the same grasp while
reachable. It schedules the next stroke from the actual reached angle, skips
already-completed waypoints, and skips navigation when already at the dock.
A pre-execution planning rejection after progress triggers release/regrasp;
a physical collision or lost contact remains a failure.

This retained-grasp policy is enabled for the wide household cabinet. Other
fixtures retain their existing stroke policy pending individual qualification.

| Same authored cabinet scene | Handle grasps to open | Opening simulation time | Final opening | Opening wall time |
| --- | ---: | ---: | ---: | ---: |
| Original `qualify_cabinet_storage_9et3nmm7` | 6 | 157.38 s | 89.3 degrees | 348.55 s |
| Optimized `qualify_cabinet_storage_c3_w5mtv` | 3 | 87.47 s | 89.3 degrees | 193.20 s |

The observed opening is 44% shorter in simulated/video time. Opening wall time also fell by 45% in this pair of runs. This is not a
claim about total generation wall time or the complete storage task. The
optimized open/close physics qualification passed: opened to 89.3 degrees and
closed to 0.098 degrees. [Open/close video](../runs/qualify_cabinet_storage_c3_w5mtv/composite.mp4).

## Pull, release, then push

`fixtures/door_push.py` selects a native panel face whose contact force produces
the requested opening torque. It closes the empty gripper **before approaching**,
measures the pad geometry to plan the contact pose, and pushes along the hinge
arc. It does not call the handle-grasp routine for the push. It verifies pad
force, actual hinge movement, the opening endpoint, and release.

The fresh public-interface opening in `qualify_cabinet_storage_j7vv4b5v`
passed: one handle grasp to approximately 42 degrees, release, one local
reposition, and a panel push to 89.3 degrees with clean withdrawal. Opening
simulation time was 58.37 seconds versus the original 157.38 seconds (63% shorter).
Closing also passed, ending at 0.098 degrees with the hand released. Opening
wall time was 138.68 seconds versus 348.55 seconds originally (60% shorter).
The earlier retained-handle comparison was 193.20 seconds.
[Combined pull/push and close video](../runs/qualify_cabinet_storage_j7vv4b5v/composite.mp4).

The wide native household cabinet now prefers this strategy, including when
reusing an older scene manifest. Other fixtures retain their existing strategy.
If both pushing approaches are rejected before reaching/contacting the panel,
it falls back to the verified retained-handle strategy. Physical execution
failures do not trigger that fallback or get relabeled successful.

An early scratch check rejects a base that obstructs the door sweep before arm
IK. Arm/panel collision checks cover the full planned swing. Wrist orientation
can change during pushing without being rigidly attached to the door.

[Short push-continuation video](../runs/cabinet_push_replay_1ms_inww/push_continuation.mp4)
is a saved-state diagnostic, not a complete episode. Its physical push reached
86.8 degrees and released; that older harness then raised a reporting KeyError
because snapshots expose `closed`, not `open`. The fresh public-interface test
above uses the actual fixture postcondition and reaches 89.3 degrees.

```bash
ROBOCASA_DIR=$PWD/external/robocasa/robocasa \
  /home/leos/.venvs/sim/bin/python scripts/qualify_composite_fixture.py \
  cabinet_storage --door-only --push-contact \
  --scene-from runs/qualify_cabinet_storage_gojfw4a5
```

Use `--handle-only` instead of `--push-contact` to compare retained handle grasps. Neither qualification
transfers objects. Both run opening and closing through the public skill interface.

All successful motion must use the passive native hinge and robot contact.
No live fixture joint assignment, added fixture actuator, or collision-mask
change is used. Scratch hinge states are only for collision prediction.

## Validation

- Original baseline: `runs/qualify_cabinet_storage_9et3nmm7/skill_calls.json`.
- Retained handle open/close: `runs/qualify_cabinet_storage_c3_w5mtv/skill_calls.json`.
- Pull/push open and close: `runs/qualify_cabinet_storage_j7vv4b5v/skill_calls.json`; all four public calls passed.
- Focused door, fixture-limit, and household setup suite: 58 tests passed.
- Added early base-sweep check: all 7 push-specific tests passed afterward.
- Push physics was qualified before the early base-sweep optimization; that
  optimization only rejects an obstructing stance earlier, without changing
  accepted motion or collision limits.

## Push to close

Fresh public-interface qualification `qualify_cabinet_storage_y9zvuo93` passed
all four calls: navigate, open, navigate, close. Closing now uses the closed,
empty gripper to push the front panel in two arcs (about 89→45→0 degrees), with
one dock per arc. It does not grasp the handle or the panel edge. The contact
point is inset from the free edge to avoid the protruding handle. Full native
collision checks remain enabled, including during approach and withdrawal.

| Closing strategy | Simulation/video time | Skill wall time | Final angle |
| --- | ---: | ---: | ---: |
| Retained handle, `j7vv4b5v` | 73.84 s | 164.05 s | 0.098° |
| Panel push, `y9zvuo93` | 54.51 s | 88.30 s | 0.271° |

This is 26% shorter simulated motion and 46% less closing wall time on the
same authored scene. The hand was released with no remaining pad contact.
It is not a measurement across all cabinet assets or a complete storage-task
success. [New open/close video](../runs/qualify_cabinet_storage_y9zvuo93/composite.mp4).

Wide native household cabinets now enable `door_push_close`, including reused
older manifests; other fixtures keep their existing policy. Set
`door_push_close: false` in a fixture binding to retain handle-based closing.
The qualification script accepts `--push-close`; `--handle-only` disables both
opening and closing pushes for a baseline comparison.

The first front-face approach (`gzmw3p8m`) was correctly rejected before contact
because the gripper would strike the handle. Moving the push point inward
passed a saved-state diagnostic and then the fresh qualification above.
The focused geometry, door-motion, fixture-limit and cabinet suite passed 48 tests.

Final suite after enabling wide-cabinet defaults: 65 tests passed.
[Closing-only video](../runs/qualify_cabinet_storage_y9zvuo93/close_door.mp4)
is an original-speed excerpt of the successful public `close_door` call.

## Fridge opening regression and staged fix

The meal rerun `composite_pilot_85c3itu1/prepare_and_serve_meal_01_original`
failed before pulling the fridge: all six handle candidates failed the
predicted 60-degree arc at 36 degrees. The fridge had retained the legacy
large-stroke routine instead of incremental docking.

Fridge bindings now use 15-degree minimum strokes with collision-checked
retained-grasp extension and released local repositioning. The same default
also applies to older manifests. No hinge actuation or tolerance relaxation
was added. `scripts/qualify_composite_fixture.py fridge --scene-from PATH`
qualifies both directions through the public interface.

Fresh qualification `qualify_fridge_b2h8xp9i` passed all four calls. Opening
took 55.71 simulated seconds / 147.33 wall seconds; closing took 91.63 /
259.89 seconds and settled at 0.020 degrees with no handle contact.
[Fridge open/close video](../runs/qualify_fridge_b2h8xp9i/composite.mp4).
This verifies the fridge only; composite progress is recorded separately.

## Base-assisted panel-push experiment — October 15, 2026

This is an isolated experiment in `scripts/compare_base_door_push.py`; production
`open_door` behavior is unchanged. The comparison reuses the same authored
wide-cabinet scene and identical initial navigation, handle pull, release,
reposition, and panel approach. Only the panel-opening strokes change.

- **Base only:** hold arm commands and translate the base along the contact arc.
  This reached about 87° but collided with the gripper's left coupler at the next
  stroke. The unchanged collision check rejected it; it is not a successful demo.
- **Coordinated base and arm:** translate the base while local IK follows the
  panel's changing orientation. Check the combined base/arm/hinge sweep in a
  scratch state before execution. Live door motion still comes from contact;
  there is no live door joint assignment, added actuator, or collision-mask change.
- A conservative 0.06 m/s displacement/duration setting passed open and close,
  but was slower in simulated motion than the existing arm-driven push.
  The cubic ramp's peak speed is higher than this nominal setting.

| Method | Opening simulation time | Result |
| --- | ---: | --- |
| Existing arm-driven push | 57.866 s | Open and close passed |
| Base only, 0.06 m/s | — | Rejected: linkage/panel collision |
| Base + wrist adjustment, 0.06 m/s | 61.156 s | Open and close passed |
| Base + wrist adjustment, 0.12 m/s | 56.836 s | Open and close passed |

Baseline: `runs/qualify_cabinet_storage_nzw5zadi`.
Rejected base-only: `runs/qualify_cabinet_storage_ql8qpv7s`.
Coordinated conservative: `runs/qualify_cabinet_storage_b0ldzzxz`.

```bash
ROBOCASA_DIR=$PWD/external/robocasa/robocasa \
  /home/leos/.venvs/sim/bin/python scripts/compare_base_door_push.py \
  cabinet_storage --door-only --push-contact --base-push --follow-arm \
  --push-speed .12 --scene-from runs/qualify_cabinet_storage_y9zvuo93
```

Omit `--base-push --follow-arm` for the baseline. Omit only `--follow-arm` to
reproduce the base-only failure. The focused door suite passed 28 tests; the
physical comparisons, rather than those unit tests, evaluate this experiment.

The faster coordinated run (`runs/qualify_cabinet_storage_72g5c9fs`) also passed
all four public calls. The opening saved **1.030 simulated seconds (1.8%)**.
The final pushing phase was about 4.48 s versus 5.52 s in the arm-driven run;
most opening time remains in preparation, approach, and repositioning.
Subsequent navigation took 13.266 s rather than 11.066 s because the base ended
elsewhere. Closing took 52.906 s rather than 54.506 s. The complete four-call
sequence saved only **0.430 simulated seconds** (158.968 versus 159.398 s).

Opening wall time in these single runs was 132.83 s versus 150.01 s. This is an
observation, not a robust speed benchmark: runs shared the host, and planner
latency varies. The conservative coordinated opening took 134.89 wall seconds.

**Conclusion:** coordinated base pushing is feasible on this cabinet, but it
is not a substantial video-duration improvement over the existing panel push.
Keep it experimental; remove the remaining redundant preparation/repositioning
before expecting a large full-skill speedup. This does not qualify other doors.

[Baseline open/close video](../runs/qualify_cabinet_storage_nzw5zadi/composite.mp4) ·
[Coordinated fast open/close video](../runs/qualify_cabinet_storage_72g5c9fs/composite.mp4).

## Faster coordinated push follow-up

The experiment launcher now accepts up to 0.30 m/s nominal push speed and
`--smooth-preparation` to use the conservative live timing layer for the
approach, handle pull, withdrawal and local navigation. These are live command
changes, not accelerated video playback or recorded-action replay.

Trials at 0.24 m/s without feedback were rejected for 17–18 mm TCP error
(the unchanged limit is 15 mm). An adaptive settling loop now allows up to
120 ms, sampled every 20 ms, only when the measured endpoint error exceeds
5 mm. Collision checking remains active throughout.

With feedback, both 0.18 and 0.24 m/s completed the panel push, but free-space
cuRobo withdrawal endpoints missed their targets by 16.4 and 21.1 mm. The
experiment now attempts one native-sweep-checked local IK correction from the
actual state for this specific panel-withdrawal tracking failure. It does not
catch collisions, alter tolerances, or retry the complete skill automatically.

Rejected trials are preserved in:

- `runs/qualify_cabinet_storage_s2tauxrf`: 0.24, no feedback.
- `runs/qualify_cabinet_storage_5t1xefzj`: 0.24, smooth preparation, no feedback.
- `runs/qualify_cabinet_storage_ezpc_2bg`: 0.24, feedback, withdrawal failed.
- `runs/qualify_cabinet_storage_g13w7n8_`: 0.18, feedback, withdrawal failed.

The targeted door/timing suite passed 36 tests before these feedback refinements.
Physical qualification of the final combination is recorded below.

### Qualified faster result

`runs/qualify_cabinet_storage_7x0z7zy0` passed all four public calls with
0.24 m/s nominal base push, wrist adjustment, feedback settling, local withdrawal
correction, and conservative smooth preparation. The final withdrawal correction
reduced TCP error from 21.1 mm to 0.984 mm in 0.410 simulated seconds. The native
15 mm rejection limit, collision checks, contact evidence, and fixture endpoint
checks were unchanged.

| Measured simulated duration | Previous base-assisted run (0.12) | New run (0.24 + smooth preparation) |
| --- | ---: | ---: |
| Panel-opening push phase | 4.480 s | 2.800 s |
| Public `open_door`, including local positioning and withdrawal | 56.836 s | 50.394 s |
| Public `close_door` | 52.906 s | 49.682 s |
| All four calls, including both navigation calls | 158.968 s | 144.450 s |

The push is **37.5% shorter**, and opening is **11.3% shorter** than the previous
base-assisted run. Compared with the original arm-driven opening (57.866 s),
opening is **12.9% shorter**. This is a fresh live physical execution, not a
retimed recording. Closing uses the existing arm-driven panel push; its timing
change comes from smooth preparation and its changed initial state, not a
base-driven closing implementation.

Opening wall time was 130.11 s versus 132.83 s in the earlier base-assisted run;
these single-run wall times are not a robust performance benchmark. This change
primarily shortens executed motion/video duration. It does not establish a
speedup for every door, other atomic skill, or randomized scene.

[New opening-only video](../runs/qualify_cabinet_storage_7x0z7zy0/open_door.mp4) ·
[Previous opening-only video](../runs/qualify_cabinet_storage_72g5c9fs/open_door.mp4) ·
[New complete open/close video](../runs/qualify_cabinet_storage_7x0z7zy0/composite.mp4).

Run this opt-in experiment (production defaults remain unchanged):

```bash
ROBOCASA_DIR=$PWD/external/robocasa/robocasa \
  /home/leos/.venvs/sim/bin/python scripts/compare_base_door_push.py \
  cabinet_storage --door-only --push-contact --base-push --follow-arm \
  --push-speed .24 --smooth-preparation \
  --scene-from runs/qualify_cabinet_storage_y9zvuo93
```

## Direct reach after the handle pull: no re-navigation

The previous base-sweep check predicted the door moving while the base stayed
stationary. That forced a dock change even when base and door could move
together. `scripts/direct_door_push.py` replaces that prediction for this opt-in
experiment with a coupled base/arm/hinge sweep. The closing check is unchanged.
A rejected direct approach fails the diagnostic instead of silently redocking.

The resulting sequence is:

**Pull handle → release → cuRobo reach to panel from the same base position →
contact → coordinated base push and wrist adjustment → withdraw.**

Fresh run `runs/qualify_cabinet_storage_tql26wo2` passed navigation, opening,
navigation, and closing. There is no tuck, navigation, or dock change between
the handle release and panel push. Actual base displacement occurs during the
contact push. The initial navigation and local positioning for the handle pull
are still present; this result does not claim zero navigation in the whole demo.

- Public opening: **34.428 s**, versus **50.394 s** in the previous faster run
  (31.7% shorter), or **57.866 s** for the original arm-driven routine
  (40.5% shorter).
- Opened to approximately **89.95°** and withdrew with no remaining pad contact.
- Closing also passed, ending at approximately **0.058°**.
- All 30 focused tests passed, including moving-base collision rejection and
  preservation of the live simulation state during coupled prediction.

[Direct opening-only video](../runs/qualify_cabinet_storage_tql26wo2/open_door.mp4) ·
[Previous opening with redocking](../runs/qualify_cabinet_storage_7x0z7zy0/open_door.mp4) ·
[Complete direct open/close run](../runs/qualify_cabinet_storage_tql26wo2/composite.mp4).

```bash
ROBOCASA_DIR=$PWD/external/robocasa/robocasa \
  /home/leos/.venvs/sim/bin/python scripts/compare_base_door_push.py \
  cabinet_storage --door-only --push-contact --base-push --follow-arm \
  --push-speed .24 --smooth-preparation --direct-push \
  --scene-from runs/qualify_cabinet_storage_y9zvuo93
```

This qualifies the direct strategy on this cabinet and authored scene. It is
still opt-in; it does not change every door or datagen default automatically.

## Shared door and drawer integration

The former prototype now runs from `fixtures/base_articulation.py` and the
shared fixture controllers, not a monkey-patched demo controller. Vertical
hinged doors prefer a direct panel push after the initial handle stroke.
A complete coupled base/arm/hinge sweep is checked before approaching. A
rejected approach falls back to retained handle motion; contact, collision,
and endpoint failures during execution still fail the skill. Horizontal hinges
retain their existing handle-driven controller.

Drawers retain their handle grasp while the base translates along the native
slide axis. The complete drawer/base sweep is checked before grasping, then
actual bilateral handle contact and slide tracking are checked during execution.
The base does not rotate. Drawer skills now have the same bounded local base
motion permission as door skills; other manipulation skills remain stationary.

Necessary released-hand dock changes use the existing forward-route navigation
executor, with minimum requested speeds of 0.30 m/s translation and 0.8 rad/s
rotation. A faster configured navigation speed is preserved. These settings are
restored after the local move, including on failure. Dock changes are skipped
when already at the requested pose. Contact motion uses a separate speed:
0.24 m/s nominal for door panel pushes and at most 0.12 m/s for drawers.

Fixture bindings may override `base_assisted_articulation`, `contact_base_speed`,
`fixture_nav_speed`, and `fixture_turn_speed`. Turning off base assistance retains
the arm-driven option. Existing public skill names and arguments are unchanged;
the catalogue now documents local drawer translation and coordinated door pushing.

Contact force is sampled every physics step. The same 0.05 N evidence threshold
is retained: sampling only at a stroke endpoint missed brief microwave panel
contact and incorrectly rejected an otherwise physically contacting push.
Room doors use native geometry refinement for conservative planner-box false
positives, with native swept and executed collision checks still enabled.

Opening-only temporary strategy parameters are restored on return or failure,
so an implicit incremental opening strategy cannot leak into a later close call.
This fixes an intermediate small-cabinet closing regression found during rollout.

### Drawer results

Same standalone drawer demo, conservative smooth timing:

| Public skill | Previous arm-only source | Shared base-assisted controller |
| --- | ---: | ---: |
| `open_drawer` | 23.514 s | 14.754 s |
| `close_drawer` | 20.272 s | 11.838 s |

Both calls and both navigation calls passed in
`runs/atomic_interface_suite_vux2h6_g/drawer`; previous source:
`runs/atomic_interface_suite_80b_g1gs/drawer`.
These are simulation/video durations, not generation wall-clock times.

The focused door, drawer, interface, and room-geometry suite passed 89 tests.
Final physical validation paths and videos follow below.


### Final physical validation and videos

[Review gallery: one public skill per video](../runs/shared_fixture_review_xw9o0ct2/README.md).
Clips retain original simulated timing; no video speedup or recorded-action replay.

| Fixture | Verified calls | Source run |
| --- | --- | --- |
| Wide cabinet | Open, close | `qualify_cabinet_storage_yfujaf9s` |
| Small cabinet | Open, close | `shared_small_cabinet_n4sid91p/cabinet-door` |
| Microwave | Open, close, Start, Stop | `qualify_microwave__ijqfxj5` |
| Fridge | Open, close | `qualify_fridge_eha68375` |
| Room door | Open left leaf, open right leaf | `qualify_room_door_k27bl4sl` |
| Drawer | Open, close | `atomic_interface_suite_vux2h6_g/drawer` |

These are representative fixture tests, not a guarantee for every asset/layout.
Room-door closing was not tested in this batch. Fridge and room-door panel
approaches remained blocked and used retained-handle fallback. Small-cabinet
opening also used fallback and was not faster than the previous controller.
The drawer gains above should not be interpreted as universal door speedups.


## Subsequent component-first optimization

The [all-atomic component audit](atomic_component_optimization.md) separates
route/posture/readiness optimization from subsequent smoothing/interpolation.
Latest wide-cabinet opening is 29.70 s and closing 41.48 s; initial navigation
is 14.76 s. [Latest cabinet clips](../runs/atomic_components_wide_cabinet_review/README.md).
These are live physical skill executions, not accelerated video or offline replay.
