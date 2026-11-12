# Qwen household component results — November 10, 2026

Development branch: `codex/qwen-capx-policy`. These are selected component trials,
not a held-out success rate or full composite evaluation. The table below is the
initial eight-case suite; subsequent qualification appears in the dated
follow-ups. See [API and setup](qwen_household_skills.md).

Latest microwave result: **continuous tilted place → release → redock → pick →
withdraw passes** in `microwave_cycle02_dynamic_map` (48.30 simulated / 111.51 wall
seconds). [Full video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_cycle02_dynamic_map/demo.mp4).
This uses calibrated skill arguments and sensor geometry; autonomous Qwen
parameter selection and the full meal composite remain separate evaluations.

| Case | Physical result and qualification | Wall seconds | Simulated seconds | Artifacts |
|---|---|---:|---:|---|
| Table placement | Pass; initial current-pose attempt fails, explicit loaded redock succeeds, then placement succeeds. | 190.0 | 32.88 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_place05/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_place05/report.json) |
| Open cabinet | Pass; approximately 90 degrees, handle released. | 154.5 | 56.92 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite02/open_cabinet/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite02/open_cabinet/report.json) |
| Cup in open drawer | Pass; Qwen raises the held-object view through redock, then places from current pose after base dock fails. | 83.1 | 16.30 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cup_drawer05/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cup_drawer05/report.json) |
| Correct stove knob | Pass; observed pot on front-left burner, only matching knob turned on. Already at a frontal dock. | 67.2 | 15.20 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_knob02/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_knob02/report.json) |
| Open fridge | Pass; Qwen continues panel contact with base push/rotation, reaches approximately 90 degrees, releases and withdraws, then finishes. | 234.1 | 93.30 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge21/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge21/report.json) |
| Open drawer | Pass; Qwen calls the skill, drawer opens 0.599 m, releases and withdraws, tool succeeds, then Qwen finishes. Checked withdrawal pose retained. | 86.7 | 33.62 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_drawer13/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_drawer13/report.json) |
| Bowl in cabinet | Fail; no validated loaded insertion/dock. First exhaustive planning attempt was manually interrupted; subsequent Qwen retries completed. | 433.0 | 3.52 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/bowl_in_cabinet/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/bowl_in_cabinet/report.json) |
| Pot on stove | Fail; placement commands completed but pot remained against the gripper, away from the requested burner. | 53.8 | 11.98 | [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/pot_on_stove/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_suite03/pot_on_stove/report.json) |

Wall time includes initialization, model calls, perception/planning and physics;
post-run video rendering is excluded. Simulated time covers the whole component
trial (including 0.4 s warmup and any retries), not just the final atomic action.
Videos show actual 1× simulated motion, not wall time.

The knob trial's actual knob call is 14.8 simulated seconds; the full trial is
15.2 s. The drawer-placement trial is 16.3 s total including view adjustment and
retry; this is not a single 16.3 s placement call. Cabinet opening remains 56.9 s
including grasp, contact motion and withdrawal; it is not timing-qualified to
the 5–15 s target. No collision disabling or changed goal tolerances were used.

## What the tests establish

- Registered non-navigation tools accept optional redock. Held-object movement
  uses observed payload geometry and retains the grasp.
- Cup insertion demonstrates recovery after a failed preparation: expose the
  support, receive a truthful base-dock failure, then plan from the current pose.
- Knob selection uses the documented seven-knob convention and image-based pot
  identification. Its success does not establish docking from a post-place view.
- Opening trials begin before grasping. Placement trials begin holding the
  object. They do not prove a full navigate → open → pick → insert sequence.
- The cup/drawer fixture is a recorded configuration reset with zero initial
  velocities; its provenance is in `runs/qwen_household_fixtures/cup_drawer_start`.
- Cabinet insertion restores an aperture measured from earlier onboard views;
  it does not use simulator fixture pose as a target.
- The stove failure demonstrates why command completion and Qwen `finish()`
  cannot replace private physical scoring. New tool feedback explicitly says
  when visual release confirmation is inconclusive.

## Code-level verification

Tests exercise handler registration, tool-result delivery, held-object state,
loaded redocking, optional current-pose placement, shared dispatcher aliases,
bounded prompt history, private support scoring and safe failure feedback.
65 interface/regression tests passed. Physical videos above are separate evidence
from these interface tests.

## Fridge opening: successful continuation

`qwen_household_fridge21` is a fresh Qwen-controlled physical pass. Qwen receives
partial-opening feedback, requests successive panel pushes, changes to
`base_rotate`, then calls `finish`. The independent scorer confirms approximately
90 degrees of opening and a released gripper. The complete trial uses nine tool
calls, 234.14 wall seconds and 93.30 simulated seconds. The video is 93.30 seconds
at 1× playback. It is not qualified against the typical 5–15 second atomic-call
target, nor does this single recorded-start trial establish robustness.

The previous 44-degree tracking failure is addressed by reacquiring neighboring
observed panel patches, retaining a consistent panel face, and holding the fitted
hinge during pushing. Intermediate feedback lets Qwen change its contact motion
without restarting the handle grasp. Planning uses observed geometry; final
fixture truth remains private scoring only.

The user-proposed withdraw → redock → recontact → push recovery remains a
separate, unimplemented fallback. This successful trial uses coordinated base
rotation while continuing panel contact; it does not validate that fallback.

## Agent-controlled cabinet redocking

`qwen_household_cabinet07` tests the November 10 request to let Qwen decide when
to redock. Loaded cabinet calls now accept a standoff and lateral offset without
requiring a prior failed attempt. A failed optional `redock=True` no longer
vetoes the requested manipulation; it remains recorded separately. Forty-two
focused regression tests passed, including returned/raised redock failures and
loaded parameter forwarding without releasing the object.

In the physical trial Qwen independently requested four docks: `(0.45, 0.0)`,
`(0.50, 0.10)`, `(0.60, 0.20)`, and `(0.40, 0.0)` metres
`(distance, lateral_offset)`. All four had no feasible planned route. Four
current-pose placement attempts also failed to find insertion paths. Final
physical success is **false**; the bowl is not released onto the shelf.
The trial took 207.96 wall seconds and 3.10 simulated seconds, eight tool calls.
This validates agent access/continuation, not successful redocking or insertion.

The Qwen cabinet profile now tries local paths without global cspace fallback;
placement attempts returned in 29.7–37.8 wall seconds in this trial. The preceding
`qwen_household_cabinet06` was manually interrupted during slow global searches
and is not counted as a completed trial. The narrower search can miss paths a
global search would find; no collision or scoring tolerances were relaxed.
Standalone backend defaults and the qualified set-table pipeline are unchanged.

- [Trial report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cabinet07/report.json)
- [Live log / completed transcript](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cabinet07.log)
- [Video, 1×](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_cabinet07/demo.mp4)

## Remaining work and earlier diagnoses

### Fridge panel continuation diagnosis

`qwen_household_fridge20` is a **failed** Qwen component trial, not a successful
opening. It verifies the new handoff: Qwen calls `open_door`, receives
`partial_open_panel_contact_ready`, then requests `continue_door` with base push,
arm swing and base rotation options. The first base-push call advances the
observed angle from about 26° to 41° in 6.06 simulated seconds.

The next call loses the narrow panel tracking patch. **44.15° is the stale
sensor estimate, not the physical final angle.** Private post-run diagnostics
show 61.34° after that call and 69.62° after the next arm attempt. Later calls
execute no motion because the stale geometry blocks planning. Overall:
204.65 wall seconds, 53.14 simulated seconds, 12 calls, no final release/pass.

- [Failure segment, 1×](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge20/panel_push_failure.mp4)
- [Full video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge20/demo.mp4)
- [Report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_household_fridge20/report.json)

Earlier failure causes corrected in code: the push flag was absent; an inner
liner appeared closer than the initially observed edge; changing from front-face
to inner-face tracking wrongly made the front face a static obstacle; one contact
IK branch had no valid approach although another branch did. Geometry comes from
RGB-D and robot calibration. Private geometry identities were inspected only to
diagnose the rejected point, never provided to the policy or execution planner.

The next tracker adds neighboring observed patches/orientation hypotheses and
keeps the hinge fitted during the pull instead of refitting transported anchors.
It reacquired about 16.3° of missed rotation from the saved failure's RGB-D.
That reconstruction check is not a fresh physical opening pass.

### Follow-up trials (November 10)

The drawer initially passed only its private fixture goal while its tool failed.
`qwen_household_drawer13` subsequently passed the complete Qwen call, including
release and withdrawal. Earlier incomplete results below remain recorded for
diagnosis, not counted as additional successes.

| Trial | Policy | Result | Wall / simulated seconds |
|---|---|---|---|
| `qwen_household_drawer10` | Fixed semantic command, no Qwen decisions | Drawer opens 0.599 m and releases; all three withdrawal paths rejected. Private goal passes, tool fails. Video rendering was interrupted after the physical trial. | 399.8 / 32.48 |
| `qwen_household_fridge07` | Fixed semantic command | Initial grasp closes empty; fresh retry has no reachable handle grasp. Panel push is not reached or qualified. | 94.1 / 12.14 |
| `qwen_household_cabinet05` | Qwen | Loaded redock executes; subsequent insertion attempts have no validated placement path. | 412.9 / 13.24 |
| `qwen_household_pot06` | Qwen | Upright cookware proposals have no validated placement path; redock has no observed route. | 163.3 / 0.40 |

The drawer withdrawal continuation probe restores the recorded released state
only for test reset. Planning receives onboard depth and robot calibration.
Straight outward, upward/outward and upward local paths all start within the
planner's 5 mm world-clearance margin (first reported gap about 1.66 mm).
This was covering-sphere overhang on `fr3_link1`: the calibrated collision-shape
box remained clear at the unchanged 5 mm margin. The Qwen drawer profile now
uses these boxes for the arm/world narrow phase, retaining native self collision
and physical collisions. The complete physical run validates the resulting
withdrawal. An unconditional final tuck was omitted; subsequent Qwen redocking
still checks its sweep with the retained posture, and arm actions plan from
measured joints. Other fixture postures and qualified set-table behavior remain
unchanged. Twenty-eight targeted regression tests passed for these changes.

The final drawer trial takes 33.62 simulated seconds including 0.4 s warmup;
its skill call takes 33.22 simulated / 77.89 wall seconds. Full-range contact
pulling and grasp/settling still exceed the 5–15 s motion target. This pass is
functional qualification, not duration qualification or a robustness estimate.
The next fridge trial selected the correct vertical door handle and reached
about 25 degrees, but panel-push preparation failed (`qwen_household_fridge10`);
it is not a completed opening.

1. Drawer robustness across other starts/handles, full-chain continuation and timing.
2. Fridge robustness across other starts, timing, and optional withdraw/redock/recontact recovery.
3. Loaded cabinet insertion and docking with adequate observed floor coverage.
4. Stove release/withdrawal that leaves cookware supported on the chosen burner.
5. Full chained trials and further contact/docking retiming before claiming all
   eight requests are physically ready or optimized.

### Cabinet alignment-only diagnostic — November 10

`runs/approach_modes_cabinet_align01` physically executed the approach from the
same recorded cabinet start. The checked arm path lowers and rotates together
into the outside-opening alignment pose; it does not require a horizontal wrist
rotation at the high carry pose first. Calibrated arm collision boxes resolve
covering-sphere overreach; native self-collision, full gripper geometry and
observed payload collision checks remain enabled.

- Fixed semantic component command, **not a Qwen policy success**.
- Alignment motion: **3.84 simulated seconds**; whole harness **4.24 simulated
  seconds / 24.47 wall seconds** (rendering follows execution).
- [Normal-speed onboard video](../runs/approach_modes_cabinet_align01/demo.mp4).
- Object remains held; insertion/release deliberately not executed. Private
  placement score is **false**, as expected. This is not a cabinet-place pass.
- Earlier `approach_modes_cabinet01`, `02`, `03` failed during planning before
  alignment moved. Their short warmup recordings do not demonstrate alignment.

### Horizontal wrist, insertion and reverse withdrawal

`runs/approach_modes_cabinet05` executed a drawer-style horizontal wrist approach,
straight Cartesian insertion, release and reverse withdrawal back to the original
arm joint posture. This is a fixed-command component diagnostic, not a Qwen
success. The **unchanged private placement score failed**: the bowl was sideways
at release and subsequently fell to the floor. Completed commands are not task
success. The wrist-mounted image alone cannot establish a bowl's world tilt;
post-run physical diagnostics confirmed the sideways orientation here.

- [Full 22.9-second video at normal speed](../runs/approach_modes_cabinet05/demo.mp4).
- Whole harness: **66.43 wall seconds**, excluding deferred rendering.
- Manipulation: **22.50 simulated seconds**, including approximately 5.9 s
  approach, 3.5 s insertion/descent, 0.8 s release, 9.2 s reverse return,
  2 s observed settling and 1 s private scoring. Exceeds the 15 s motion target;
  further timing qualification remains. No video acceleration.
- Previous `approach_modes_cabinet04` rejected all 24 insertion candidates before
  executing (86.37 wall seconds, 0.4 simulated warmup seconds).

Horizontal insertion and a horizontal wrist are different constraints. Changing
wrist orientation rotates the attached bowl. The horizontal-wrist experiment is
therefore an **explicit option**, not the automatic default. Reproduce its motion
choice with `--component-front-horizontal`; `front_wrist="object_level"` remains
the default, although the current partial-surface orientation estimator is not
physically qualified for this cabinet. A successful level-bowl cabinet placement
and return cycle is still outstanding.

That trial reversed the complete approach after release, including an initial
upward motion. The correction below supersedes that withdrawal sequence. Cartesian insertion uses robot-calibration IK along the
line, 50 Hz velocity/acceleration-limited retiming and swept collision checks.
48 focused unit tests passed after these changes; they do not replace the failed
physical placement score.


### Horizontal-first withdrawal and inset placement — November 10

The user's video diagnosis was confirmed in private contact replay: in run 05,
the left finger pad still touched the released bowl when lifting began 1.1 s
after release began. The bowl lifted with the finger and eventually fell. Tilt
alone was an incomplete explanation.

The new front-entry sequence is **release → horizontal exit at release height →
restore the alignment posture outside the opening → reverse the free approach**.
No lifting or wrist rotation is requested while exiting. The exit is planned
from observed aperture geometry and the calibrated gripper envelope; private
contact flags are used only for this post-run audit, not execution.

Run 07 executed that sequence but still failed: the bowl had not been supported
by the shelf and continued riding on the finger. Horizontal-wrist proposals now
require the estimated object footprint to be inside the observed opening, with
3 cm additional clearance. This preserves support/obstacle tests. The resulting
candidate search found a deeper supported placement on candidate 26.

**`runs/approach_modes_cabinet09`: private cabinet placement score PASS.**

- [Complete normal-speed onboard video](../runs/approach_modes_cabinet09/demo.mp4).
- [Private contact audit](../runs/approach_modes_cabinet09/withdrawal_contact_audit.json).
- Bowl separated from the hand at the first recorded clear sample, **0.9 s after
  release began**. Lift began at **5.9 s**; at that point the bowl touched the
  cabinet shelf, not the gripper. Audit samples are 10 Hz.
- **24.68 simulated seconds / 90.15 wall seconds** for the component harness;
  rendering happened afterward. Skill execution was 24.28 simulated seconds.
  This exceeds the 15-second target and is not timing-qualified: full checked
  insertion/exit/return and release/settling remain; candidate search dominates
  much of wall time.
- Fixed semantic command with `front_wrist="horizontal"`, not an autonomous Qwen
  decision and not a full composite pass. Target coordinates and paths came from
  observations; recorded state supplied only the private test reset.
- Test candidate limit was 96, success used candidate 26. Harness default is now
  48 (previously 24 stopped before this candidate). Normal HouseholdSession still
  permits its existing exhaustive default when no limit is supplied.
- The release backend's old bowl-on-table diagnostic remains false for cabinet
  placement; the harness's unchanged **cabinet** semantic scorer is the relevant
  result. Visual settling confirmation was inconclusive; that was not relabeled
  as a visual success.
- 49 focused unit tests passed, including horizontal-before-lift ordering and
  the observed-opening footprint constraint. Robustness across other starts,
  cabinet assets and autonomous Qwen decisions remains unqualified.

### Qwen-ranked docking trials — November 10, 2026

Opt-in `--qwen-docking` keeps the qualified sensor set-table pipeline unchanged.
The docking subpolicy sees navigation RGB, wrist RGB and the **actual observed
SLAM grid** with numbered, sensor-generated endpoint candidates. Qwen selects
IDs; it never supplies world coordinates. Endpoint checks and the retained
robot/payload swept planner still authorize execution. If its selected route
fails, a shared multi-goal A* may select another endpoint; logs distinguish that
fallback from executing Qwen's first choice.

The component trial's semantic command order is fixed (`--policy skill`);
Qwen chooses the dock. This is **not** an autonomous end-to-end Qwen pass.
Initial unrestricted Qwen trials (`qwen_dock_*01`) were stopped because they
attempted reopening an open fridge / placing before docking; do not count them
as successful docking experiments.

| Trial | Physical result | Episode wall / simulated seconds |
|---|---|---|
| Previous `qwen_household_place05` | Table placement pass after an unsuccessful current-pose attempt | 190.0 / 32.88 |
| `qwen_dock_table02` | Table placement pass; Qwen selected shown dock 5; checked direct route | 127.08 / 28.22 |
| `qwen_dock_table03` | Table placement pass; prepared dock preserved across default approach context | 102.29 / 26.62 |
| `qwen_dock_fridge02` | No base motion: Qwen selected shown dock 3, but routes vetoed for missing floor evidence | 14.65 / 0.40 |
| `qwen_dock_fridge04_candidate1` | Human-requested candidate-1 geometry; still no base motion with restored past floor/footprints | 8.07 / 0.40 |

[Latest successful table video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_dock_table03/demo.mp4).
[Fridge map actually shown to Qwen](/nobackup/le/cross-episode-qwen-capx/runs/qwen_dock_fridge02/redock_000/qwen/candidate_map.png).

These are individual trials, not a success-rate estimate or controlled evidence
that Qwen alone caused the timing improvement. The new trial prescribes redock
before place, avoiding the previous failed placement attempt. Default approach
options previously invalidated a compatible prepared dock because `None` and
the explicit defaults compared unequal; those representations now normalize to
the same options. A true approach change still invalidates the prepared plan.
Table redocking still exceeds the 15-second simulated target (~18 seconds),
using the existing 0.08 m/s follower. The user accepted the table behavior;
no faster follower was installed in this change. Perception, docking, release
and final private semantic scoring remain enabled. Video rendering is deferred
and excluded from the reported episode wall times.

Fridge context recovery reconstructs only preceding onboard frames, uses RGB-D
SLAM poses, and filters both floor snapshots and support history by the reset
timestamp. It does not import simulator world poses or future evidence. Even
this did not remove all floor-visibility gaps; the subsequent user decision is
recorded below and in the skill documentation.

`qwen_dock_fridge05_candidate1` validates removal of the unknown-floor veto:
**redock PASS**, 25.60 wall seconds / 21.46 simulated seconds, target distance
1.481 → 0.559 m. The requested candidate-1 geometry was selected by the user,
not Qwen. No past floor context was needed in this trial. Observed-obstacle
swept checks and physical collisions remained active. The local move retained
the current arm posture; turning/settling kept it above the 15-second target.

The following **pickup failed** during arm planning (`Aligned insertion IK
unreachable`); no grasp was executed. Full component: 77.09 wall seconds /
21.86 simulated seconds, private pickup score false. Do not describe this as a
successful fridge-pick demonstration.
[Docking and failed pickup-attempt video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_dock_fridge05_candidate1/demo.mp4).
44 focused tests passed, including evidence that unknown-floor traversal leaves
map evidence unchanged and continues to reject an observed obstacle.

`fridge_old_dock06` compares the previous automatic five-candidate docking
without `--qwen-docking`, with the same post-retreat reset and unknown-floor
traversal retained. It moved but reached `local_motion_budget`: 30.02 simulated
seconds / 34.71 wall seconds, final target distance 0.763 m versus requested
0.55 m. Subsequent pickup also failed; full component 54.54 wall seconds /
30.42 simulated seconds. This is not a successful rollback. The earlier
human-selected candidate-1 redock remains the physically successful docking
example. 57 focused tests passed after these changes.

### Shared horizontal fridge alignment qualification

`fridge_horizontal_align08` physically passed **alignment only** with Qwen
choosing the dock. The robot reached approximately 0.761 m from the perceived
bottle; the arm then aligned outside the opening with a horizontal approach
axis and level closing line. Alignment took 4.36 simulated seconds; complete
component including navigation/perception: 52.24 wall / 24.12 simulated seconds.
`front_horizontal=true`, `alignment_success=true`; pickup success remains
false because the diagnostic deliberately leaves the gripper open and does not
insert/grasp. Do not count this as a fridge pickup pass.
[Alignment video](/nobackup/le/cross-episode-qwen-capx/runs/fridge_horizontal_align08/demo.mp4).
The subsequent full pickup trial is `fridge_horizontal_pick09`.


### Fridge pickup diagnosis and aligned-start pass — November 10

`fridge_horizontal_pick09` and `pick11` docked successfully but failed before
arm insertion. The tucked wrist saw only the bottle's lower end, resulting in
low grasp proposals whose horizontal hand geometry collided near the shelf.
`pick10_current` separately produced no usable M2T2 proposals from its restored
view; this is not evidence that the bottle was absent.

Aligning the wrist exposed the full bottle, but `pick12_aligned` revealed a
second bug: the densest observed horizontal level was the shelf **above** the
bottle (1.665 m), misidentified as its support. Confined pickup now selects a
broad observed surface below the target (1.070 m in this fixture). `pick13`
then planned insertion but rejected withdrawal because its initial supported
pose was incorrectly checked as already free of the shelf. The support-breaking
phase now permits the starting support contact during a monotone, fixed-attitude
vertical lift; side/overhead geometry remains checked, and full obstacle
clearance is required at the lifted endpoint and throughout withdrawal.

`fridge_horizontal_pick14_aligned` **passed physical pickup** from the previously
executed aligned posture: 14.20 simulated seconds for pickup, 14.60 seconds
including harness settling; 31.86 wall seconds before deferred rendering.
The bottle lifted 0.176 m and remained physically grasped. This is an isolated
arm test, not yet an end-to-end docking result.
[Pickup video](/nobackup/le/cross-episode-qwen-capx/runs/fridge_horizontal_pick14_aligned/demo.mp4).

The complete sequence **passed** in `fridge_horizontal_pick15_full`:
Qwen ranks sensor-derived docks, then the skill aligns, observes again, grasps,
lifts off the shelf and withdraws. Fixed semantic skill order is used in this
component harness; do not call it autonomous high-level Qwen planning.


Full recorded-start component result: **71.12 wall seconds before deferred
rendering, 43.76 simulated seconds**, bottle lift 0.175 m, sustained physical
hold verified by the private scorer. Qwen selected shown dock 4 (~0.75 m
standoff); sensor A* / checked route execution reached it. Redocking took
20.06 simulated / 25.90 wall seconds. The one public `pick` call, including its
alignment/re-observation, took 23.30 simulated / 41.18 wall seconds. Neither
call is yet within the typical 5–15 s duration target. Docking retains route
turning/settling; pickup retains the observation-changing alignment, grasp
closure verification, support-breaking lift and confined withdrawal. No video
acceleration or altered physical success criterion was used.

[Full docking and pickup video](/nobackup/le/cross-episode-qwen-capx/runs/fridge_horizontal_pick15_full/demo.mp4)
([live log](/nobackup/le/cross-episode-qwen-capx/runs/fridge_horizontal_pick15_full.log)).
65 focused tests passed. One complete successful fridge component trial does
not establish success across unseen fridge layouts. This starts with the
fridge already open; opening/closing and meal-preparation completion are not
part of this trial. Table placement was not modified by these fixes.

Measured pickup phases in the full trial: initial alignment 6.26 s (0.10 s
settling); post-observation approach 6.58 s (0.10 s settling); insertion 3.12 s
(0.10 s settling); gripper closure 0.32 s; lift/withdrawal 6.02 s (0.16 s
settling); final hold observation 1.00 s. The additional post-observation
approach is a remaining optimization opportunity; the first camera view and
fresh grasp view produced different opening/entry estimates. It has not yet
been optimized away or claimed to be necessary in every scene.


### Stove and basket component tests — November 10, evening

`basket_place02_view` **passed physical containment and release** through
`execute_skill('place', {'receptacle':'purple basket','region':'interior',
'approach_mode':'top_down'})`. It starts holding the second bottle, with one
bottle already in the basket. A checked 1.44 s view-clearing motion exposes
its bottom, then the basket-specific backend plans release/withdrawal.
Total: **12.06 simulated s for the skill; 12.46 simulated / 91.68 wall s for
the component**, excluding deferred rendering from wall time. Most of the
wall cost is perception and 30 rejected candidate paths before a usable pose.
The private scorer requires the initially held object's physical floor support,
full projected containment, and no hand contact. Rim placement is rejected.
[Basket video](/nobackup/le/cross-episode-qwen-capx/runs/basket_place02_view/demo.mp4).

`basket_place01` could not see the floor from the initial held posture and
executed no release. Stove trials `stove_place07`–`10` remain failures at the
time of this entry: upright placement is unreachable from the existing dock;
loaded docking exposed self/held-object evidence in its frozen local map.
No stove-placement success is inferred from command completion.

`stove_place16_corner` **passed** the unchanged private physical score:
initially held cookware released, physically supported by the stove, no hand
contact, and within the requested front-left burner zone. The public `place`
call took **10.64 simulated / 64.53 wall seconds**. The component including
harness initialization/settling took **11.04 simulated / 68.63 wall seconds**,
excluding deferred rendering. Video: **11.1 s at 1×**.
[Stove video](/nobackup/le/cross-episode-qwen-capx/runs/stove_place16_corner/demo.mp4)
([log](/nobackup/le/cross-episode-qwen-capx/runs/stove_place16_corner.log)).

The successful stove run retains the same episode's pre-grasp observed cookware
shape. The live pickup already supplies that shape, but the mid-episode test
fixture had been rebuilding an incomplete surface behind the fingers. The new
`--held-observation` test option restores RGB-D points in the calibrated tool
frame; it does not restore a target coordinate or trajectory. Fresh Molmo
pointing selects the burner after a checked **1.42 s** view-clearing arm motion.
The clarified query distinguishes front/rear and left/right cooking zones.
The native measured-plane settling fallback supplies placement proposals;
forced upright candidate replacement is no longer the interface default.
Eight unreachable candidates were rejected before the ninth succeeded.

Failure history remains relevant: `13` used native proposals but incomplete
attachment geometry and an occluded burner view; `14` could not execute its
view posture with that partial geometry; `15` restored the attachment and
physically placed/released the cookware, but missed the unchanged 9 cm named
burner criterion (9.2 cm). `16` passes that criterion. It is a single successful
recorded-start component, not evidence of robust unseen-scene targeting. The
navigation and wrist burner points still disagree; view-consistency refinement
remains a limitation. Do not present this as full heating/serving success or
as a qualified far-start loaded redock. Earlier redock trials `08`–`12` failed.

Both successful components retain checked free approach/lowering, measured
finger release, reverse withdrawal, and the backend's bounded visual settling
and private diagnostic hold. Visual settling remained inconclusive for the
stove; the public result says so, and private physical scoring independently
passes. No collision disabling, success-threshold relaxation, or video speedup
was used. Both **skill** times meet the typical 5–15 s target; their wall times
remain dominated by perception and candidate planning. **48 focused tests pass**
for interface dispatch, approach behavior, payload handling and component scoring.

Reproduce from this worktree with the existing model-service endpoints:

```bash
export SENSOR_MODEL_ENDPOINTS="$(cat /nobackup/le/cross-episode-sensors/runs/five_task_sensor_20261110_clear_candidate22/endpoints.json)"
CUDA_VISIBLE_DEVICES=2 bash scripts/test_qwen_household.sh \
  --case pot_on_stove --policy skill --turns 1 --tool-budget 16 \
  --recording /nobackup/le/cross-episode-sensors/runs/household_fix_20261110_burner_settle79/place/state_recording \
  --held-observation /nobackup/le/cross-episode-sensors/runs/household_fix_20261110_burner_settle79/place/input.npz \
  --frame-index 1 --max-place-candidates 24 --render \
  --output runs/stove_component_retest
CUDA_VISIBLE_DEVICES=1 bash scripts/test_qwen_household.sh \
  --case grocery_in_basket --policy skill --turns 1 --tool-budget 16 \
  --recording /nobackup/le/cross-episode-sensors/runs/household_fix_20261110_grocery_release59/place/state_recording \
  --frame-index 1 --render --output runs/basket_component_retest
```

Use fresh output directories. These commands use a fixed semantic skill call;
the models/perception/motion execute live. They do not demonstrate autonomous
Qwen high-level planning. The earlier table/fridge pickup executors were not
changed by these stove/basket fixes.

### Microwave placement and Qwen pickup choices — November 10, evening

`microwave_place09_point` **passes placement**: the initially held food
container is released and physically supported inside the microwave, with the
hand withdrawn. The unchanged private appliance scorer passes; the legacy
bowl-on-table diagnostic in the placement backend is not this task's score.
The final wrist image shows the container on the turntable.

- Skill: **51.72 simulated / 63.64 wall seconds**.
- Component: **52.12 simulated / 68.15 wall seconds**, excluding deferred render.
- [Placement video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place09_point/demo.mp4).
- [Placement log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place09_point.log).

This is a recorded-start component from an already docked, held-object state.
The earlier sensor-opening-memory approach `microwave_place04_aperture` passed
base docking (32.62 simulated seconds) but its placement failed. These separate
passes are **not a complete meal-preparation or continuous approach/place/pick
qualification**. Placement exceeds the 5–15 second target: recorded phases are
approximately 6.1 s approach, 16.0 s insertion/lowering, 0.9 s release, 25.8 s
withdrawal, plus visual settling/private diagnostic time. The existing aligned
translation retiming is conservative; speed optimization is still unresolved.
No video acceleration or relaxed task score was used.

The context in `runs/microwave_component_context` comes from the same source
episode: pre-grasp RGB-D points transformed by calibrated tool FK, and an
observed door-opening aperture transformed by recorded SLAM poses. Its JSON
provenance files identify those sources. These are sensor memory, not fixture
truth or authored parking coordinates.

Pickup is **not yet qualified**. Qwen run `microwave_pick09_qwen_direct` really
chose learned front entry, horizontal front entry, direct top-down, and then
redocking/retries. It failed the physical pickup score: no grasped/lifted food
container. Total: **170.45 wall / 13.66 simulated seconds**, five Qwen turns.
[Qwen comparison video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick09_qwen_direct/demo.mp4)
([log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick09_qwen_direct.log)).
The completed closer-dock calibrated-palm trial `microwave_pick10_qwen_palm`
also fails (132.48 wall / 6.88 simulated seconds). Some downward contacts are
kinematically unreachable; other candidates intersect the appliance/counter.
A kinematic-only probe found some downward endpoints reachable with the target
10 cm closer, but the physical base route to that closer position was blocked.
This is not evidence that moving closer is executable or that a grasp succeeds.

Earlier failures exposed implementation issues: floor points excluded by the
appliance body mask; red mask overlays misleading the verifier; support
selection applied only to one approach mode; learned orientations overwritten
by a horizontal-only branch; and an extra learned-wrist alignment cycle hiding
the target from the wrist camera. The new options/dispatch address those issues.
A transient target-memory initialization error in `microwave_pick06_qwen_support`
was also corrected. Preserve these failed results; command completion and a
nonempty-looking gripper are not private task success.

The farther-start Qwen comparison `microwave_pick11_qwen_far` also **fails**:
90.66 wall / 7.48 simulated seconds, four turns. Its actual transcript tries
`front_entry/learned`, `front_entry/horizontal`, and `top_down`. Its last call
requests `redock_distance=0.3`, outside the documented 0.35–1.5 m range, and
executes no base motion. The video is 7.5 s because most candidate planning
fails before motion; wall-time reasoning/planning is not simulated video time.
[Latest comparison video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick11_qwen_far/demo.mp4)
([log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick11_qwen_far.log)).

Final cleanup keeps horizontal and learned orientations as explicit alternatives,
rather than silently appending learned poses to a horizontal request. Fine local
voxels and the palm-box narrow phase apply to shallow confined objects; tall
fridge-bottle and ordinary table grasp geometry retain their existing defaults.
**73 focused tests pass**, including Qwen option propagation, original alignment
behavior, calibrated collision rejection, voxel enclosure, and appliance scores.
This unit coverage is not physical pickup success. No processes from these
microwave component trials are left generating further attempts.

### Top-down-first rerun — November 10

`microwave_pick12_qwen_top_first` confirms that Qwen's first pickup call is
`approach_mode="top_down"`. It **fails before grasp motion**: the candidate
approaches have no collision-clear IK solution at the restored placement dock.
Qwen then requests `redock(..., lateral_offset=0.0)` for an empty-hand microwave
approach, which is unsupported by that API, and tries learned front entry;
that attempt also fails. It calls `finish()` in the same generated cell rather
than inspecting feedback in another turn. Private pickup success is **false**.
Total **53.70 wall / 0.40 simulated seconds**; the simulated time is harness
settling, not a completed pick. The short video therefore contains no grasp
motion. The requested order is implemented, but neither a successful pickup
nor successful adaptive recovery is established by this run.

[Live/final log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick12_qwen_top_first.log)
[Video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick12_qwen_top_first/demo.mp4)

### Qwen microwave rim calibration — November 10

`microwave_pick13_qwen_calibration` completes eight model turns, **pickup false**,
224.85 wall / 0.40 simulated seconds. First top-down, then explicit 40°, 55° and
20° tilted-rim trials. Later automatic-redock retries fail before base movement
against sensed obstacles and repeat earlier grasps. No physical grasp executes;
the 0.40 s is initialization. At 55°, some endpoint solutions exist but the
full entry path is rejected. This is not successful recovery. The next test
retains up to twelve compact history turns for microwave calibration, so early
failed attempts remain visible, and prompts a separate distance/offset adjustment
after repeated infeasibility. Other tasks keep their four-turn history default.

User follow-up: test centered microwave placement before retrieval. The previous
front-entry policy explicitly ranked nearest supported contacts. New opt-in
`placement_preference="center"` ranks already-supported candidates within 4 cm
of the sensed center. For microwave this uses a verified wrist turntable point
rather than averaging conflicting navigation/wrist points. It neither teleports
the object nor changes the general semantic appliance score. Physical centered
placement and subsequent retrieval require separate qualification.

### Centered placement follow-up — November 10

The previous placement preferred the nearest supported contact. Testing a
centered goal exposed additional issues; none of these runs qualifies a full
place-and-retrieve cycle:

| Trial | Physical result | Wall / simulated seconds | Finding |
|---|---|---:|---|
| `microwave_place10_center` | Failed before release | 170.98 / 0.82 | Centered insertion unreachable from old dock. |
| `microwave_place11_center_dock` | Interrupted, failed | 60.20 / 1.24 | Standalone microwave redock incorrectly dispatched to generic table perception. |
| `microwave_place12_center_aperture` | Interrupted after dock failure | See report | Correct aperture backend, but unobserved floor veto prevented local docking. |
| `microwave_place13_center_local` | Failed physical score | 117.32 / 50.26 | Dock succeeded; release routine finished, but container followed the withdrawing gripper and fell. Do not count tool completion as success. |
| `microwave_place14_center_upright` | Failed before release | 74.00 / 0.82 | Upright-object entry paths unreachable/colliding. |
| `microwave_place15_center_branch` | Failed before release | 52.73 / 0.82 | Measured center plus 180° wrist roll: contact/entry arm-counter conflicts remain. |

The local dock in trial 13 took 17.56 simulated / 31.82 wall seconds. Placement
commands took 32.30 simulated / 81.12 wall seconds, with alignment, insertion,
release and horizontal withdrawal retained. These exceed the typical 5–15 s
atomic target and are **not timing-qualified**. The initial full-body turns and
contact/withdrawal duration remain optimization gaps; videos are rendered at 1×.

Standalone redock now recognizes a remembered appliance opening even outside a
pick/place approach context. Qwen's local loaded aperture docking permits unknown
floor like its empty-hand local docking, while retaining observed obstacle and
whole robot/payload swept checks. Global navigation defaults are unchanged.
Tests explicitly verify unknown-floor allowance does not disable obstacles.

Pointing at an interior floor does not guarantee the pointing pixel is the
center. Trial 15 instead estimates the middle of measured horizontal support
extents at the pointed floor height, in the observed opening frame, and ranks
supported contacts within 4 cm. This estimates cavity-floor center, not an exact
circle fit to the turntable. Centered front entry can seed alignment from checked
contact IK branches; placement still plans its complete horizontal withdrawal.
`wrist_roll_degrees` exposes horizontal placement roll for calibrated attempts;
180° in trial 15 came from the previous successful placement, not fixture truth.

`microwave_place16_center_side` also **fails placement** (67.64 wall / 17.38
simulated seconds). The 12 cm sideways aperture dock succeeds (30.72 wall /
16.56 simulated seconds), but centered 180°-roll contact poses conflict with the
observed counter/adjacent fixture. No release executes. All trials in this
section are finished; no background retries remain. **92 focused tests pass**,
plus three unknown-floor/obstacle regressions; this does not qualify physical
centered placement or retrieval. No successful center-place→pick video exists.

- [Executed failed placement/withdrawal video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place13_center_local/demo.mp4)
- [Sideways dock, insertion rejected](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place16_center_side/demo.mp4)
- [Latest trial log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place16_center_side.log)

The next physical issue is the release/withdrawal geometry and a reachable
center-insertion posture. Do not label these failed trials successful based on
`release_commands_completed`, a selected center point, or unit-test coverage.
The earlier `microwave_place09_point` remains a successful **non-centered**
placement only. The latest changes are isolated on `codex/qwen-capx-policy`;
other household tasks have not been physically requalified with them.

### Tilted centered placement reproduced — November 10

`microwave_place21_tilt_clearance` **passes private physical scoring**: the
initially held food container is released and supported inside the microwave.
Final wrist imagery confirms it remains on the turntable after withdrawal.
This is a live component from a saved held-object start, using the sensor skill
interface with prescribed calibration arguments (`--policy skill`), not a
Qwen-selected policy or a full composite success.

- [Successful 1× video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place21_tilt_clearance/demo.mp4)
- [Physical report](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place21_tilt_clearance/report.json)
- [Final wrist image](/nobackup/le/cross-episode-qwen-capx/runs/microwave_place21_tilt_clearance/final_wrist.png)

The chosen candidate uses a 40° object tilt, a 5 cm release clearance, and a
contact 1.18 cm from the estimated support center. This candidate distance is
not a measurement of the object's final center. Alignment, insertion, release,
horizontal withdrawal and posture restoration execute through physics with
collisions retained. The skill took **73.67 wall / 24.70 simulated seconds**;
including setup, **78.42 wall / 25.10 simulated seconds**. Contact and checked
alignment/withdrawal remain above the 5–15 s target; this is not timing-qualified.

The earlier upright estimator could select a visible side wall as the object's
bottom. Tilted shallow-object placement now estimates the thin axis of the
observed held cloud, transforms object and tool together, and checks calibrated
closed-finger insertion plus open-finger withdrawal geometry. Release-clearance
choices are interleaved so a finite candidate budget does not exhaust itself on
the lowest clearance alone. No oracle object pose or target coordinate is used.
Legacy non-tilted placement defaults are retained.

`microwave_pick14_tilt55_low` still failed preflight from the older placement
(71.75 wall / 0.40 simulated seconds); no grasp executed. Placement success does
not establish pickup success. A top-down pickup from trial 21's exact final state
was tested separately in `microwave_pick15_from_tilt`. Focused regression tests:
**58 passed** (approach geometry, household dispatch and calibrated arm geometry).

Pickup follow-ups from trial 21's final state all finished without a successful
grasp. Top-down (`microwave_pick15_from_tilt`, 31.05 wall seconds) had no clear
approach IK. A 55° tilted rim (`microwave_pick16_from_tilt55`, 61.39 s) had
floor/linkage conflicts or no validated entry path. A 40° tilt with contact 1 cm
above the observed rim (`microwave_pick17_from_tilt40`, 28.43 s) had no reachable
contact. Each advanced only 0.40 simulated seconds during setup. These failures
do not invalidate the placement pass, but **the complete place→pick cycle is
still unqualified**. No background tests remain from this sequence.

### Pickup after tilted placement — November 10 follow-up

`microwave_pick21_side_rim` **passes** from the final state of the successful
tilted placement. A 12.7 cm sensor-based redock retained the arm posture and took
2.28 simulated / 4.66 wall seconds. The selected grasp was an observed side-rim
hypothesis with a 55° downward tilt, level jaws, a 5 mm rim-height offset and
4 cm raised, sloped final approach. This avoids both the low insertion path and
the jaw tilt that drove a finger toward the floor.

- [Successful pickup video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick21_side_rim/demo.mp4)
- [Physical pickup report](/nobackup/le/cross-episode-qwen-capx/runs/microwave_pick21_side_rim/report.json)

The container rose **16.6 cm** and remained in contact with the gripper throughout
the private hold verification. The complete skill including redock took
**49.07 wall / 23.20 simulated seconds**; setup-inclusive totals were
54.45 wall / 23.60 simulated seconds. It still exceeds the 5–15 s target because
checked approach, contact settling and confined withdrawal are retained.
No video acceleration or collision disabling was used.

Earlier attempts 18–20 remain failed: a 0.65 m requested dock was obstructed,
changing wrist yaw alone was insufficient, and sloped entry with the old contacts
still had linkage/path conflicts. Side-rim hypotheses and the feasible 0.8 m
requested dock produced the successful component. These are sensor-geometry
refinements, not new M2T2 weights. **60 focused regression tests pass.**

Continuous place→pick qualification is recorded separately as
`microwave_cycle01_tilt`; the two component successes alone do not establish its
outcome.

`microwave_cycle01_tilt` failed (140.64 wall / 25.10 simulated seconds): placement
passed, but the continuous handoff retained the old held-container patch in the
navigation map. It blocked the pickup redock, leaving the arm too far from the
new side-rim contacts. The placement backend had fused unfiltered RGB-D before
applying its observed-payload mask. It now fuses the filtered packet and uses
`invalidate_moved_surface` to invalidate only old held-surface evidence not
supported by current static observations. Cells become unknown, not fabricated
free space. The existing static-surface protection regression passes.

`microwave_cycle02_dynamic_map` reruns both actions continuously with that map
handoff fix. The test starts from the same saved held-object state, but does not
restore simulator state between placing and picking. Its private score requires
both phases and matching released/picked object identities. The test's semantic
commands are fixed calibration arguments; Qwen does not select them in this run.

Reproduce the continuous diagnostic in the existing sensor environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
SENSOR_MODEL_ENDPOINTS="$(cat /nobackup/le/cross-episode-sensors/runs/five_task_sensor_20261110_clear_candidate22/endpoints.json)" \
bash scripts/test_qwen_household.sh \
  --case place_microwave --policy skill \
  --recording runs/microwave_place13_center_local/001_placement/aperture_dock/state_recording \
  --frame-index -1 \
  --observed-aperture runs/microwave_component_context/center_dock_aperture.json \
  --held-observation runs/microwave_component_context/held.npz \
  --component-front-wrist object_level \
  --component-options '{"object_tilt_degrees":40,"entry_height":0}' \
  --pickup-after-place --render --turns 1 --tool-budget 24 \
  --max-place-candidates 32 --output runs/microwave_cycle_new
```

Use a fresh output directory and currently running service endpoints. The saved
start and same-episode sensor memories are component-test artifacts, not inputs
available to an autonomous policy in a new scene.

**Final cycle result: PASS.** `microwave_cycle02_dynamic_map` privately verifies
release onto microwave support, followed by a retained grasp of the same object
outside the opening. The object lifted 17.7 cm during retrieval. There is no
intermediate simulator reset or oracle-coordinate injection.

| Phase | Wall seconds | Simulated seconds |
|---|---:|---:|
| Place, release and withdraw | 75.20 | 24.70 |
| Pick including redock | 31.88 | 23.20 |
| Total including setup | 111.51 | 48.30 |

Redock retained the posture and took 2.56 simulated seconds. Pickup itself took
20.64 simulated seconds: approach 7.30 s, final insertion 4.40 s, finger settling
0.46 s, loaded lift/withdrawal 7.48 s and hold verification 1.00 s. The approach,
insertion and withdrawal include measured arm-settling times of 0.10, 0.14 and
0.16 s. Required contact/hold checks remain; motion is not yet qualified to the
typical 5–15 s atomic target. Rendering occurs after execution at 1× speed.

- [Continuous place-and-pick video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_cycle02_dynamic_map/demo.mp4)
- [Physical score and timings](/nobackup/le/cross-episode-qwen-capx/runs/microwave_cycle02_dynamic_map/report.json)
- [Live-run log](/nobackup/le/cross-episode-qwen-capx/runs/microwave_cycle02_dynamic_map.log)

The first continuous failure remains recorded. The successful rerun confirms the
map handoff fix in addition to both manipulation skills. No other composite is
claimed requalified by this test.

## Full-task regression and meal integration — November 11, 2026

`set_table_after_tilt02/episode` passed full semantic scoring: two requested
bowls and two cups on the table, no newly added wrong contents, empty hand.
Execution took **1807.18 wall seconds (30 min 7 s)** before deferred rendering.
This is a fresh, continuous two-room run, not a replay of the component test.

The additional household recipe previously still dispatched microwave placement
and retrieval to older generic backends. `runtime/household_components.py` now
connects the qualified `HouseholdSession` skills to the non-set-table recipe:
centered 40-degree object-tilt placement, 55-degree side-rim pickup with the
tested elevated entry, and the aligned fridge pickup with Qwen dock ranking.
Every aperture and payload comes from this episode's sensor observations;
no successful component recording or geometry is loaded. Pickup hands its
actual execution output to the existing carry code. Placement retains its
updated map, including moved-payload invalidation. The set-table dispatch is
unchanged. Fixture calls use the household contact profile in noninteractive
mode so the fixed recipe executes its full opening command.

`meal_after_tilt01` was stopped during exploration after inspection identified
an empty local SLAM-view history at the first manipulation checkpoint. The
bridge now seeds that view with the existing tracker's current observation;
it does not create another tracker. Six bridge tests pass, including failure
propagation, semantic-memory continuity, and the pose-history regression.
The existing focused regression suite also passed.

`meal_after_tilt02` was the fresh full prepare-and-serve-meal test. **Failed at
microwave placement; stopped during subsequent recipe execution to fix it.** It started from the original initial two-room
fine-tuning scene, with inter-room doors open, and includes microwave heating,
food retrieval, fridge pickup, and serving. Final semantic scoring remains
private and unchanged. Rendering is deferred until execution ends.

- [Meal live log](/nobackup/le/cross-episode-qwen-capx/runs/meal_after_tilt02/live.log)
- [Set-table report](/nobackup/le/cross-episode-qwen-capx/runs/set_table_after_tilt02/episode/report.json)
- [Meal implementation snapshot hashes](/nobackup/le/cross-episode-qwen-capx/runs/meal_after_tilt02/implementation_manifest.json)

### Meal microwave docking recovery — November 11, 2026

The successful earlier component started at a saved centered dock. The full
recipe instead used a multi-goal shortest base route that selected a lateral
dock: its sensed support center was about 35 cm sideways versus 8 cm in the
earlier success. Identical tilted placement options were therefore insufficient.
The first center-only retry (`meal_place_center01`) still selected a 0.70 m
standoff and could not reach insertion. A 0.60 m centered retry (`center02`)
exposed another bug: aperture docking extruded the remembered microwave leaf
from floor height to twice its handle height, falsely blocking the robot base.

`observed_open_panel_points` now supplies actual depth returns near the
sensor-estimated opened leaf. It never invents a floor-height panel; the prior
observed obstacle map, full robot/payload swept checks, and physics collisions
remain active. The meal bridge now uses the tested loaded `redock` at 0.60 m
standoff and zero lateral offset before the existing 40-degree tilted insertion.
A failed placement preserves held state and cannot invoke empty-hand return.

`meal_place_center03` **passed** from the exact failed placement state: actual
redocking, insertion, release and withdrawal, with private verification that
the initially held food is supported inside the microwave. Total **109.51 wall
seconds / 47.36 simulated seconds**, including redocking (24.26 simulated
seconds) and placement (22.70 simulated seconds). These motions exceed the
typical atomic-duration target and are not timing-qualified. Retained phases
include checked base turns, loaded view preparation, insertion, release/hold
verification and confined horizontal withdrawal. No video acceleration or
success-rule relaxation is used. **87 focused tests passed.**

The failed-state component fixture reanchors this same episode's RGB-D hinge
estimate with its recorded SLAM poses; its payload comes from its recorded
pre-release depth surface. It does not reuse another episode's fixture pose.
See `runs/meal_failed_place_context/provenance.json`.

- [Successful redock and placement video](/nobackup/le/cross-episode-qwen-capx/runs/meal_place_center03/demo.mp4)
- [Physical result](/nobackup/le/cross-episode-qwen-capx/runs/meal_place_center03/report.json)
- [Follow-up continuous place/pick log](/nobackup/le/cross-episode-qwen-capx/runs/meal_place_center04_cycle.log)

The follow-up `meal_place_center04_cycle` also **passed**: redock, insert,
release, withdraw, then retrieve the same food in one continuous simulation.
Private scoring confirms the same object was first supported in the microwave
and then lifted/held. **140.45 wall seconds / 67.46 simulated seconds** before
rendering. Pickup lifted the food 18.37 cm. Its local redock took 1.20 simulated
seconds; placement took 22.74 simulated seconds. This is still a component
qualification, not a completed heating-and-serving task.

- [Continuous recovery, placement and pickup video](/nobackup/le/cross-episode-qwen-capx/runs/meal_place_center04_cycle/demo.mp4)
- [Continuous private score](/nobackup/le/cross-episode-qwen-capx/runs/meal_place_center04_cycle/report.json)

The full integration rerun `meal_after_tilt03` failed before reaching the new
placement operation: a redundant return to the pre-opening handle dock timed
out, leaving a poor held-object view/arm-clearance pose. Subsequent posture
failures and a missing navigation-dock dereference were cascading failures.
It was stopped for repair; it is not a successful meal example.

Microwave navigation now skips that old handle-view return: the placement
component owns its loaded aperture redock. Fixture calls without a successful
navigation report reacquire from current onboard images instead of dereferencing
`None` or retaining a stale handle hint. Other fixture return paths and the
qualified set-table dispatch are unchanged.

`meal_carry_place01` starts at **this full run's actual carry endpoint**, before
the redundant return. It passed loaded redocking, placement, release, withdrawal,
and retrieval of the same object continuously: **142.29 wall seconds / 81.90
simulated seconds**, before rendering. Its payload is reconstructed from that
episode's pregrasp depth and executed commanded grasp TCP; the aperture uses
that episode's RGB-D hinge fit and SLAM poses. No object truth enters execution.
Motion remains over the typical 5–15 s atomic target (loaded turning/clearance
and confined insertion/withdrawal); this is physical qualification, not timing
qualification. The focused household regression suite passed 55 tests.

- [Carry-to-placement and retrieval video](/nobackup/le/cross-episode-qwen-capx/runs/meal_carry_place01/demo.mp4)
- [Carry handoff physical result](/nobackup/le/cross-episode-qwen-capx/runs/meal_carry_place01/report.json)
- [Full rerun live log](/nobackup/le/cross-episode-qwen-capx/runs/meal_after_tilt04/live.log)

`meal_after_tilt04` is the fresh full integration rerun. Consult its log and
final report for whole-meal status; the component passes above do not establish
heating, closing appliances, drink retrieval, or serving success.


## Appliance closing: observed opposite-face dock (November 11, 2026)

Opening and closing require different contact views. Closing now anchors the
base approach to the observed swung-open panel, using its stored RGB-D hinge
estimate transformed by SLAM. The closing pointer identifies the open free
edge; the microwave keypad is not a closing-handle reference. A freshly
reacquired open-handle cloud must not be rotated through the opening angle a
second time when building the closing collision scene.

`microwave_close04_contact_frame` passed physical closing, release and checked
stow from the actual failed meal-run pre-close state. The hinge moved from
−1.570813 rad to 0.000028 rad. The skill took **154.41 wall seconds / 87.88
simulated seconds**: docking 39.76 s, handle acquisition 22.16 s, contact and
withdrawal 79.03 s, and stow 13.34 s (wall). This exceeds the atomic timing target;
it is a component pass, not a successful complete meal.

- [Successful microwave closing video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_close04_contact_frame/demo.mp4)
- [Result](/nobackup/le/cross-episode-qwen-capx/runs/microwave_close04_contact_frame/report.json)

Panel push closing is a separate **experimental opt-in** (`--panel-close`),
reusing the observed panel-push contact backend and native arm/base control.
It does not regrasp the handle. The first microwave test moved the door about
31.5° toward closed but failed at an unreachable contact continuation; the
first fridge test could not find a collision-clear push approach beside its
protruding handle. Neither is a successful close. Follow-up tests move the
contact inboard onto the panel and share hinge rotation with the base. Do not
make this the default or infer full-pipeline success from the experiment flag.
The focused closing/contact/household regression suite passes **96 tests**.


### Panel push-close qualification

`microwave_close09_push_frontal` **passes physical close, release and stow**.
The unchanged private scorer measured the hinge at 0.000029 rad after starting
at −1.570813 rad. It uses a frontal contact candidate, a preflight of the first
coordinated pushing sweep, the observed floor map, and base yaw following.
No handle grasp was required. Skill timing is **158.89 wall / 87.74 simulated
seconds**; whole harness before rendering is 166.36 wall / 88.14 simulated
seconds. This is comparable to, not faster than, the 154.41-wall-second grasp
close. Docking took 40.88 wall seconds, panel preparation/push about 104.26 s,
and checked stow 11.75 s. Contact motion and docking remain over the atomic
5–15 simulated-second target; no video speedup is used.

- [Successful push-close video](/nobackup/le/cross-episode-qwen-capx/runs/microwave_close09_push_frontal/demo.mp4)
- [Physical result](/nobackup/le/cross-episode-qwen-capx/runs/microwave_close09_push_frontal/report.json)

The fridge push tests remain unsuccessful: `fridge_close09_push_frontal`
rejects all candidate pushing continuations from the saved opening endpoint.
Do not count it as a closing or meal-composite success. The experiment remains
opt-in and the proven microwave grasp-close path remains available. Regression
coverage including contact retreat and new push-continuation checks: **109 passed**.


## Grocery sorting integration — November 11, 2026

`grocery_sort_current01` started the full two-room, initially open-door recipe
from `unpack_and_sort_groceries_000001`. Exploration and the first drink's
pickup, carry, basket placement and return stages passed. The second pickup
failed before executing a grasp: its navigation target was at height 0.769 m,
and the local measured crop contained only the approximately 0.77 m table.
The remaining bottle is visible in the recorded onboard views. This is a
pointing/mask grounding failure, not evidence that the bottle is ungraspable.

The failure also exposed stale recipe context: the next carry tried to reuse
`transfer_001/pick/physics` after pickup 3 failed. New attempts now replace the
pickup reference and clear the previous held/carry context before acquisition.
No failed attempt may inherit another object's successful pickup.

Grocery-only mask refinement now estimates support height from neighboring
RGB-D pixels and retains the connected surface above it. Table-only masks are
rejected; flat box tops above the table remain valid. The ordinary set-table
pointing hook returns the original mask unchanged. The real saved bottle mask
retains 691 of its original 763 pixels. Focused perception/recipe tests: 23 pass.

The full attempt was interrupted for repair during subsequent snack navigation;
there is no full-task success result. `grocery_second_drink_recovery02` tests
fresh perception, docking and production pickup from the failed second-drink
state with its prior observed floor restored. A component result does not count
as a successful complete sorting episode.

- [Original live log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current01/live.log)
- [Failure navigation view](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current01/episode/transfer_003/navigation/diagnostic_navigation.png)
- [Recovery log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_drink_recovery02.log)


`grocery_second_drink_recovery03` passed corrected perception, redocking and
production pickup from the failed second-drink pose. Private scoring measured
24.11 cm lift of the remaining drink; the sensed gripper was occupied. Pickup
motion took 6.66 simulated seconds. The earlier `recovery02` exposed a component
harness bug (attempting to update a frozen map); that harness now preserves its
frozen map through the pickup handoff. `grocery_sort_current02` is the full fresh
integration rerun; consult its report for complete-task success, not this
component result.

- [Successful recovery result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_drink_recovery03/report.json)
- [Full grocery rerun live log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current02/live.log)

### Qwen basket redocking — November 11

`grocery_sort_current02` finished unsuccessfully. Its second drink was picked
and carried, but `transfer_004/place_dock` still used automatic dock selection,
bypassing Qwen. It failed before moving. The basket backend now calls the shared
Qwen ranking and route-planning functions by default, with 0.50–0.75 m candidate
standoffs and the same local unknown-floor policy as Qwen redocking.

`grocery_qwen_dock01` restored that failed physical state. Qwen ranked candidate
5 first; A* reached that exact candidate without geometric fallback, and private
contact diagnostics confirmed that the drink remained held. Docking passed in
**28.72 simulated seconds / 77.35 wall seconds**, including a 4.69-second Qwen
selection call. The arm remained raised throughout; the loaded route's turns
and settling still exceed the 15-second typical-call target. No video speedup
or collision disabling was used. Focused docking/handoff tests: **28 passed**.

The subsequent basket placement failed before motion: zero learned or measured
footprint placement proposals survived. This is a docking component pass, not
a placement or complete grocery-task pass. The regression rebuilt local 3D
obstacles from current RGB-D and restored recorded floor coverage; it could not
restore the unexported full episode occupancy. Persistent-map integration has
not yet been requalified with this change.

- [Qwen choice and docking result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_qwen_dock01/dock/report.json)
- [Docking video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_qwen_dock01/dock/rollout.mp4)
- [Component live log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_qwen_dock01.log)

### Basket interior refresh after Qwen docking

The failed placement treated the pre-redock floor memory as complete and skipped
fresh container detection. That sparse 237-point support patch left every tested
footprint overlapping the first bottle or the rim. The refreshed wrist view
contained additional visible free floor. Basket placement now reacquires the
observed interior after redocking and merges it with the previous support
measurements. Collision, footprint, release and semantic scoring rules are
unchanged; no new surface is inferred from simulator geometry.

`grocery_qwen_place02` restored the successful dock and passed placement of the
second drink into the partially occupied purple basket. There were 331 learned
proposals, none surviving the conservative circular support filter, followed by
65 measured-footprint fallback proposals. The ninth tested physical approach
passed. This success uses the geometric fallback, not a learned M2T2 placement.
Private scoring confirmed release and support inside the basket. Motion took
14.96 simulated seconds and the stage took 40.04 wall seconds; visual settling
also passed. The legacy bowl/table diagnostic in the script is irrelevant for
groceries; the dedicated basket scorer is in the component root report.

- [Placement result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_qwen_place02/report.json)
- [Placement video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_qwen_place02/place/rollout.mp4)
- [Full integration rerun log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current03/live.log)

`grocery_sort_current03` is a fresh full-episode test. Its final task score,
when available, determines full qualification; the component pass does not.

### Full-run regressions and placement-aware docking

`grocery_sort_current03` completed unsuccessfully (14/17 public calls passed).
The first drink reached a Qwen-selected dock about 0.76 m from its target but
all arm placement paths failed. The second drink's placement passed. The first
snack's basket-floor observation failed and the backend then loaded a nonexistent
input cloud. The last snack failed to confirm the orange basket from its carry
view. These failures preceded final scoring; they were not a stricter final
pose requirement.

Corrections in this branch:

- Basket docks now use native arm/payload approach-and-lower planning at each
  Qwen-ranked candidate before moving. Robot calibration, observed scene points,
  observed payload and estimated SLAM transforms supply this lookahead. The
  base-map planner retains floor support; the arm lookahead excludes measured
  ground-plane points to avoid treating normal base/floor contact as an obstacle.
- Basket perception points at the visible interior before selecting a segmented
  region, preventing a small outer-wall crop from excluding the floor. A failed
  view is physically changed and reacquired; no missing input cloud is fabricated.
- Long narrow observed payloads receive upright-first footprint proposals;
  basket releases try 5 cm clearance before higher drops. Flat packages retain
  the existing orientation search. Other receptacles keep their defaults.
- Failed placement preserves the held-object handoff instead of declaring an
  empty-hand return. The private task scorer remains unchanged.

Physical component results (not a complete-task success):

| Recorded case | Result | Placement simulated time | Placement wall time |
| --- | --- | ---: | ---: |
| First drink, corrected dock and upright placement (`grocery_first_drink_upright02`) | Released and supported inside purple basket | 27.98 s | 90.69 s |
| Previously successful second drink (`grocery_second_drink_regression01`) | Regression passed inside occupied purple basket | 14.80 s | 67.01 s |
| First snack, full basket skill (`grocery_orange_skill03`) | Observation, dock and private placement scoring passed | 7.30 s | 32.98 s |

The first drink remains over the typical 15-second motion target: its validated
arm trajectory and settling take longer. Planning retries are included in wall
time. The second-drink and first-drink visual settling signals were inconclusive;
their separate physical basket support/containment scoring passed. Geometric
placement fallbacks are identified as such, not learned M2T2 successes.

- [First drink result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_first_drink_upright02/report.json)
- [First drink video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_first_drink_upright02/place/rollout.mp4)
- [Second drink regression](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_drink_regression01/report.json)
- [Orange basket skill result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_orange_skill03/report.json)
- [Orange basket placement video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_orange_skill03/place/rollout.mp4)

### Basket-floor and approach correction, November 11

The closer-view retry exposed a second perception error: a point on the rim
seeded a thin height slice of the wall. Fitting a plane after slicing could
misclassify that band as horizontal support. `basket_floor_points` now measures
normals in the original RGB-D neighborhood, within the bounded basket instance
mask, before selecting the lowest observed horizontal floor patches. The failing
orange view had floor at approximately 0.932 m and rim at 1.014 m; these numbers
are diagnostic observations, not thresholds or scene constants in execution.

The basket skill first tries the existing view. If no measured placement
footprint fits, it approaches for a closer view before raising the payload.
`approach_receptacle` merges swept-clear collinear segments and consecutive
rotations, retains necessary turns, and derives its allowance from route length
and turns. The former last-snack approach timed out after 40 simulated seconds;
the corrected approach completed in 21.70 s, removing 13 redundant waypoints.
No collisions, support criteria, or private semantic checks were disabled.

Current physical regressions:

| Case | Approach | Clearance | Qwen dock | Place | Private result |
| --- | ---: | ---: | ---: | ---: | --- |
| First drink, original carry endpoint (`grocery_first_drink_skill03`) | 11.40 s | 3.02 s | 10.52 s | 27.82 s | Released inside purple basket |
| Last snack, original carry endpoint (`grocery_last_snack04`) | 21.70 s | 6.70 s | 11.62 s | 7.36 s | Released inside orange basket |
| Second drink, occupied basket (`grocery_second_drink_regression02`) | Existing dock | — | Existing dock | 16.02 s | Regression passed |

Times are simulated, with physical execution at 50 Hz. Entire basket calls remain
over the typical 15-second target: these include view approach, retained-payload
clearance, redocking and placement. The bottle's arm path and settling alone take
27.82 s. Last-snack docking took 110.66 wall seconds, largely placement-feasibility
planning; first-drink placement took 95.08 wall seconds including rejected plans.
This is a correctness improvement, not full timing qualification. Visual settling
was inconclusive for the bottles; unchanged private support/containment scoring
passed. Last-snack visual settling also passed. The first-snack component passed
earlier; complete-task qualification must still test both snacks together.

83 targeted tests passed, covering handoffs, missing/sparse support recovery,
rim-versus-floor geometry, Qwen docking, route compaction and component scoring.
These tests do not replace the physical regressions.

- [First drink video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_first_drink_skill03/place/rollout.mp4)
- [Last snack video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_last_snack04/place/rollout.mp4)
- [Occupied-basket regression](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_drink_regression02/report.json)
- [Fresh full grocery run](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current04/live.log)

`grocery_sort_current04` started from the original scene through the public skill
interface. Exploration, first pickup, carry, view recovery and docking passed.
The first placement failed: its fallback regenerated contacts around 0.783 m
instead of preserving the approximately 0.759 m contact validated by docking.
The run was stopped after that failure for a handoff repair; it is not a complete
successful episode. Its recordings and implementation hashes are retained.

The next correction retains the preflight contact, orientation and release height,
reanchors them with estimated SLAM poses, and prioritizes that proposal after
fresh footprint/obstacle verification. Native arm/payload planning is still
required. Geometric fallback contacts also receive the dock-target preference
(previously ordering happened before fallback generation). Qualification starts
from the exact failed `transfer_002/place` recording in `grocery_dock_handoff01`.
Its legacy witness is recovered from saved estimated dock poses, not simulator
world poses. `grocery_dock_handoff01` passed private containment and visual settling from that
exact failed placement state. The exact preflight witness was rejected by the
fresh arm plan; the third proposal, prioritized near the retained contact,
succeeded. Previously all 200 proposals failed. Placement took 26.04 simulated
seconds and 39.39 wall seconds. This is a component result, not full-task success.

- [Handoff repair video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_dock_handoff01/place/rollout.mp4)
- [Handoff repair result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_dock_handoff01/report.json)
- [Full rerun 05 live log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current05/live.log)

85 targeted tests pass. Failed observation attempts remain in stage history and
are explicitly linked to a successful same-call recovery; this prevents the
wrapper from mislabeling a recovered skill. Full rerun 05 passed a basket placement using its retained dock witness on the
first placement attempt, then passed return motion. It is nevertheless **not a
full success**: an earlier pickup stopped on `Gripper did not settle`, causing
its dependent carry/place to fail. The run was stopped after confirming the
later basket transfer so the pickup could be repaired separately.

Private diagnostic inspection showed opposing pads contacting the bottle while
finger angles oscillated by about 0.0006 rad. The previous counter required ten
consecutive per-step changes below 0.0002 rad and kept restarting. The added
household backend now uses `FingerStabilityWindow`: ten encoder samples, total
range ≤0.001 rad and half-window mean drift ≤0.0002 rad. Continued closure,
large oscillations and invalid readings fail readiness; lift/retention and task
scoring remain unchanged. This does not modify the qualified set-table pickup.

`grocery_pick_readiness01` replayed the failed pickup start with recorded onboard
observations and fresh local RGB-D. It passed physical pickup: closure readiness
in 0.26 s, total 9.88 simulated seconds, bottle lift 0.242 m, retained throughout
all 50 private verification samples. No simulator contacts enter readiness.

- [Pickup repair report](/nobackup/le/cross-episode-qwen-capx/runs/grocery_pick_readiness01/report.json)
- [Original pickup failure](/nobackup/le/cross-episode-qwen-capx/runs/grocery_sort_current05/first_pick_failure.mp4)


## Approach before manipulation — November 11, 2026

The first Qwen dock now uses only a visible RGB-D target and the observed
navigation map. It does not test an arm grasp orientation, placement footprint
or hypothetical placement trajectory before moving. Manipulation reconstructs
from fresh onboard views after arrival. Old distant basket surface points are
not promoted to release support. Cross-room navigation and the non-Qwen
qualified set-table dispatch remain unchanged.

| Test | Actual result | Wall time | Simulated time | Evidence |
| --- | --- | ---: | ---: | --- |
| Qwen dock then bowl pickup | **Physical pickup passed**, refreshed proposals after docking | 45.70 s total; 14.40 s dock | 18.32 s total; 6.76 s dock | [Video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_approach_pick02/demo.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_approach_pick02/report.json) |
| Production basket first dock | **Dock passed and bottle retained**; later placement failed: fresh floor not recovered | 54.49 s dock | 19.42 s dock | [Dock video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_approach_first03/docking.mp4), [report](/nobackup/le/cross-episode-qwen-capx/runs/grocery_approach_first03/report.json) |

The basket's semantic-point observation took 1.50 seconds. Its 19.42-second
loaded dock exceeds the typical atomic time target due to retained conservative
translation/turn speeds and terminal settling; no video acceleration or weaker
physics checks were used. The prior 424.9-second dock failure spent time testing
hypothetical placement IK without moving. The new production path makes zero
such pre-dock arm-planning calls.

The earlier pickup trial `qwen_approach_pick01` docked at 0.76 m and failed
arm planning. Clarifying the ranking prompt produced a 0.66 m dock and the
physical pickup pass above. Basket retries `grocery_approach_first01` and `02`
also docked successfully but did not place. These results qualify the changed
approach ordering, not a successful complete grocery episode or all fixture types.

The final table regression `qwen_approach_table01` passed physical release and
support scoring: [video](/nobackup/le/cross-episode-qwen-capx/runs/qwen_approach_table01/demo.mp4),
[report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_approach_table01/report.json).
It took 161.76 s wall / 49.84 s simulated in total. Docking took 102.52 s wall
and about 39.72 s simulated; placement took 50.81 s wall / 9.72 s simulated.
This changed Qwen path is not speed-qualified: lattice travel/turning and terminal
settling exceed the docking target. The non-Qwen set-table dispatch was preserved.

Validation: 69 focused tests passed, including sparse-target approach without
a release footprint, unknown-floor versus observed-obstacle handling, fresh
manipulation dispatch after redocking, and preserving failed-dock reasons rather
than cascading into an invalid placement call. Syntax checks and diff checks
passed. Cabinet, drawer, fridge, microwave and burner physical cycles were not
rerun for this approach-order change; shared-path code coverage is not physical
qualification of those cases.

### Frontal basket dock and unobstructed reconstruction — November 11

`grocery_frontal_view02` passed physical placement of the held bottle into the
purple basket. The private scorer confirmed release and support inside the
basket. This is a recorded-state component test, not a completed grocery episode.

The dock estimates the counter's front from observed RGB-D vertical points and
faces that front. After arrival, a projected target patch revealed robot
occlusion (66%). A collision-checked lateral arm motion uncovered the basket;
fresh perception then recovered the interior floor. Placement perception also
rejects support patches far from the observed target. Neither floor reconstruction
nor hypothetical placement IK gates the initial dock.

| Component | Wall time | Simulated time | Result |
| --- | ---: | ---: | --- |
| Frontal loaded dock | 59.73 s | 21.72 s | Passed; bottle retained |
| Lateral arm clearance | — | 1.60 s | Passed |
| Reconstruction, planning and placement | 103.74 s | 7.66 s | Released and supported inside basket |

Placement succeeded with the measured-footprint geometric fallback, not a
learned M2T2 placement proposal alone. The 622 support candidates still contain
some rim points; the reconstruction figure shows their full height range.
Docking remains above the typical 5–15-second target, and placement proposal
search remains a wall-time bottleneck. No video acceleration was used.

- [Before/after onboard view](/nobackup/le/cross-episode-qwen-capx/runs/grocery_frontal_view02/visuals/before_after.png)
- [Actual reconstruction inputs](/nobackup/le/cross-episode-qwen-capx/runs/grocery_frontal_view02/visuals/basket_reconstruction.png)
- [Dock, uncover and place video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_frontal_view02/frontal_dock_and_place.mp4)
- [Physical result](/nobackup/le/cross-episode-qwen-capx/runs/grocery_frontal_view02/report.json)

The preceding `grocery_frontal_view01` passed docking but was interrupted after
perception selected an unrelated support surface; it is not a placement success.
Focused regression tests for frontage, visibility, handoff, Qwen docking and
basket floor extraction: 68 passed. Other fixtures and the full grocery sequence
have not been physically requalified by this component test.

### Second drink: interior-based base refinement — November 11

The full `grocery_sort_current07` run failed after the first placement. The
second drink was picked and carried, but its three placement attempts failed
without arm execution. Recovery then lost destination memory and later calls
cascaded into missing handoffs. It is not a full-task success.

The basket skill was missing the requested post-arrival base refinement: it
only moved the arm aside and repeated arm planning. It now reconstructs the
interior after arrival, physically redocks toward that measured interior when
off-center or distant, and reacquires geometry after moving. Frontal candidates
use the exact measured normal instead of a 15-degree yaw lattice.

`grocery_second_refined01` starts from that second drink's recorded carry state,
with the first drink already present. All five component stages passed and
private scoring confirmed release and basket support of the second drink.
The close-range anchor was initially 12.7 cm lateral; after redocking it was
64.7 cm ahead and 3.3 mm lateral in estimated base coordinates.

| Stage | Wall seconds | Simulated seconds |
| --- | ---: | ---: |
| Initial dock | 61.18 | 21.12 |
| Fresh interior observation | 9.85 | 0 |
| Physical interior refinement | 80.46 | 17.02 |
| Placement planning and execution | 118.49 | 7.10 |

The refined dock retained the loaded posture and grasp. Route turns/settling
still exceed the typical atomic target; placement required 100 candidate
attempts. This fixes the missing base correction, not the remaining runtime
cost or the full recipe's failure-continuation bug. No physics checks or scoring
were weakened. Eighteen focused tests passed, including post-arrival refinement,
already-centered handling, and exact-front candidate geometry.

- [Retry log](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_refined01.log)
- [Physical score](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_refined01/report.json)
- [Base correction and placement video](/nobackup/le/cross-episode-qwen-capx/runs/grocery_second_refined01/refine_and_place.mp4)


### Full-task facade pickup regression — November 11

`qwen_pipeline_pick01` resets to the failed second-drink navigation state from
`grocery_sort_current08`, then calls the same `full_task_interface` factory and
generic pickup backend as the full recipe. Fresh local observation corrected
the empty-table target. Qwen docking passed (16.75 wall / 14.76 simulated
seconds); pickup passed (41.7 wall seconds). Total public call: 58.68 wall
seconds, including redocking. This is a component regression, not a completed
sorting episode. Forty-six dispatch/handoff tests passed, covering all five
semantic recipes, navigation passthrough, and persisted redock feedback.

- [Component log](/nobackup/le/cross-episode-qwen-capx/runs/qwen_pipeline_pick01.log)
- [Component report](/nobackup/le/cross-episode-qwen-capx/runs/qwen_pipeline_pick01/report.json)
# Grocery visual vocabulary — November 11, 2026

Run `grocery_sort_current09` passed both drinks into the purple basket but
failed snacks (final task success false, 1348.8 wall seconds). Its snack props
look like plain blue rectangular blocks, not recognizable food packaging.

`perception/target_language.py` now expands the public `packaged snack` category
into a visual block/box description at model inputs. Navigation pointing,
One-Ring, local redocking, pickup pointing, and Qwen dock ranking share this
description. Semantic labels, object identities, scoring, saved scenes and
physical assets are unchanged. Other task targets and direct drink queries
retain their previous wording.

Recorded-image comparison: `runs/grocery_visual_prompt_check01/report.json`.
On the same first wrist image, the old query returned no points and the new
query returned two points on the blocks. Both queries found two candidates in
the first navigation view. **The new query also returned false candidates in
the final doorway view with no snack visible.** Alternative shape/location
wording did not remove this problem (`alternatives.json`). This is a prompt
improvement on the positive view, not a qualified perception or full-task fix.
No physical rerun was performed for this change. The missing-target handoff
error and erroneous approach in run 09 remain separate unresolved issues.


### Follow-up: navigation and physical block pickup pass

`runs/qwen_snack_navigation02` restored the first frame of run 09's snack
search as a private test reset, then called the recorded navigation command
and the full recipe's Qwen pickup facade. The calibrated robot-bounds filter
removed an occluding-arm point previously tracked as a nearby snack. This
uses RGB-D plus known robot geometry/encoders, without full-image masking or
simulator object identities. It is scoped to the snack category.

- Navigation passed: 36.3 wall seconds.
- Qwen redock passed: 7.85 wall seconds, 0.78 simulated seconds; final target
  distance 0.537 m. Retained arm posture passed the existing swept check.
- Pickup passed both execution and private physical scoring: 40.8 wall seconds,
  10.28 simulated seconds, sustained held-object observations.
- 52 focused regression tests passed, including missing-target handoff and
  rejecting robot points while retaining off-robot targets.

The earlier direct-redock-from-far test `qwen_snack_pick01` was interrupted
without a successful grasp at 0.76 m standoff. `qwen_snack_navigation01` was
interrupted after identifying its self-targeting problem. Neither is a pass.
The complete task rerun `runs/grocery_sort_current10` failed on the second
drink placement and was interrupted during the later snack-pick search. It is
not a complete-task pass. The checkpoint/target handoff reports a missing observed target
as a failed pickup instead of raising `KeyError` or claiming pickup success.

### November 11: loaded tuck and set-table target continuity

The Qwen basket path now finishes its base dock, physically tucks the loaded
arm, waits for measured joint settling, then obtains fresh RGB-D and plans the
placement. `tuck_payload_before_place` preserves the actual gripper command;
its arm and observed-payload sweep are checked against onboard geometry and
executed at 50 Hz. It does not use the empty-hand stow routine, which opens the
gripper. Cross-room navigation and the legacy M2T2 set-table path are unchanged.

For Qwen bowl/cup pickup, navigation's selected observed point is transformed
between estimated SLAM frames and associated with fresh points from both
cameras. A farther same-category object cannot silently replace it. A useful
existing facing dock is retained; otherwise Qwen chooses a checked local dock
for the associated target. This fixes the observed switch from the nearer blue
bowl to the harder gray bowl. It is not a general counter-edge ranking model.

| Focused physical retry | Result | Simulated time | Wall time |
| --- | --- | ---: | ---: |
| Second set-table bowl: local dock + pick | Sustained physical grasp; blue bowl lifted 0.272 m | 1.68 + 7.66 s | 27.48 s |
| Second grocery drink: loaded tuck + place | Released inside purple basket, supported by its floor | 2.34 + 9.14 s | Placement alone 105.29 s |

These tests privately reset the recorded failed states, then use the production
sensor/controller functions. They are component passes, not complete-task
qualification. Basket planning rejected 79 candidates before succeeding; wall
time remains a limitation. The successful proposal used the observed-footprint
geometry fallback, not a learned M2T2 proposal. The basket-specific private
score passed; the script's legacy bowl-on-table diagnostic is inapplicable.

- [Successful bowl pickup](../runs/set_table_bowl_identity01/successful_pick.mp4)
- [Loaded tuck and successful basket placement](../runs/basket_loaded_tuck01/successful_place.mp4)
- [Bowl component report](../runs/set_table_bowl_identity01/episode/report.json)
- [Basket private physical score](../runs/basket_loaded_tuck01/episode/report.json)

Complete reruns with these fixes: `runs/set_table_qwen_current02` and
`runs/grocery_sort_current11`. Consult their timestamped live logs and final
reports for status; the component results above do not predict full success.

### November 11: keep the mapped table identity through placement

Full set-table run 02 placed both bowls, then failed before the first cup's
placement dock. On a replay of the failed camera frame, Molmo pointed at the
robot wrist; FastSAM selected the wrist patch and Qwen accepted its category as
table despite describing a circular object. The geometric support filter found
no tabletop. This was a redundant-recognition failure after successful carry.

`scripts/test_m2t2_place.py` now reads the confirmed dining-table landmark from
the same semantic memory used for cross-room navigation. `mapped_support.py`
associates fresh RGB-D with that landmark's observed bounds, or with a local
search neighborhood when only its anchor was recorded. The neighborhood is
not a claimed free surface: only measured horizontal depth becomes support.
Robot and held-object pixels remain excluded; scene obstacles and occupied
placement footprints still enter motion planning. With a known, unambiguous
table identity, this path calls no Molmo, FastSAM or Qwen recognition. M2T2
placement proposals and optional Qwen geometric dock ranking remain separate.
Unknown receptacles retain visual discovery. This does not claim that every
basket/appliance already exists in the exploration map.

`runs/mapped_table_place01` replayed the failed first-cup carry endpoint using
the earlier exploration landmark and recorded estimated SLAM transforms:

- Fresh known-table observation: passed, 3.86 wall seconds; 11,937 navigation
  support pixels and 17,365 wrist support pixels, zero recognition calls.
- Local docking: passed, 58.67 wall seconds.
- Cup placement: passed execution and private released-support scoring,
  26.15 wall seconds / 6.74 simulated seconds.
- [Dock and successful cup placement](../runs/mapped_table_place01/successful_place.mp4)

This is a component qualification. Full rerun: `runs/set_table_qwen_current03`.
The recipe now preserves a pending grasp after failed placement rather than
running empty-hand stow or silently replacing it with another pickup attempt.

Basket recovery now repeats interior-based base docking after uncovering a
new floor patch, before attempting placement again. Initial refinement observes
after the loaded tuck. `runs/basket_redock_after_tuck01` is its recorded-state
production-skill test; its first observations still selected a rim fragment,
so those early placement attempts failed. Do not count them as successes.
The test finished **unsuccessfully**: recovery exposed a floor patch at 0.733 m
forward and redocked to 0.659 m, retaining the physical grasp. The final fresh
view generated zero learned and zero measured-footprint placement candidates;
the second drink was not released into the basket. Closer docking alone has
not solved this case. The earlier isolated tuck/place pass is not evidence of
full grocery-task reliability.
