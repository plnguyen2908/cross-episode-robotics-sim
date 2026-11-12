# Qwen household manipulation and loaded redocking

Branch: `codex/qwen-capx-policy`. This extends the Qwen/CaP-X experiment, not
the oracle interface or the qualified set-table pipeline. Implementation is in
`cross_episode_sim/code_policy/household_session.py`.

## Available calls

Qwen receives the actual schemas and current navigation/wrist RGB images.
Both physical cameras remain RGB-D; the sensor backends use metric depth.

| Request | Atomic call | Local motion |
|---|---|---|
| Place on a table | `place(receptacle="dining table")` | Current-pose placement; optional loaded redock |
| Open fridge | `open_door(target="fridge")` | Handle grasp, observed hinge motion, base/arm contact, release/withdraw |
| Open cabinet | `open_door(target="kitchen storage cabinet")` | Same observed door backend, including panel push |
| Open drawer | `open_drawer(target="kitchen drawer")` | Observed slide direction, handle grasp, pull, release/withdraw |
| Put cup in drawer | `place(receptacle="kitchen drawer", region="interior")` | Loaded docking when requested, observed interior support, release |
| Put bowl in cabinet | `place(receptacle="kitchen storage cabinet", region="interior")` | Retains the aperture observed during opening; optional loaded aperture docking |
| Put pot on stove | `place(receptacle="front left stove burner")` | Observed grate/burner support, optional loaded docking |
| Twist pot's knob | `rotate_knob(target="front left stove burner knob", state="on")` | Observed control, M2T2 grasp, rotate, release; also accepts `off` |

These spellings are aliases for `execute_skill(name, arguments)`, not extra
task-specific atomic functions. Both `place(receptacle="dining table")` and
`place({"receptacle":"dining table"})` dispatch through the same interface. Close-door, close-drawer and push-button handlers
are registered too, but are not qualified by an open-only test.

All registered non-navigation manipulation skills accept `redock=True`,
`redock_target`, and optionally `redock_distance`. Loaded cabinet docking accepts a chosen standoff; other loaded placement
backends still choose standoff from observed payload/support geometry. Explicit `redock()` uses `distance`,
not `redock_distance` (the latter belongs to `execute_skill` arguments).

```python
execute_skill("pick", {"target": "cup", "redock": True})
navigate(target="cup", destination="kitchen drawer")
execute_skill("place", {
    "receptacle": "kitchen drawer", "region": "interior", "redock": True,
})
```

Qwen may redock whenever its current views suggest it, without waiting for a
failed placement or approval. For a held bowl and an observed cabinet aperture:

```python
redock(target="kitchen storage cabinet", purpose="place",
       distance=0.45, lateral_offset=-0.12)
place(receptacle="kitchen storage cabinet", region="interior")
```

`distance` is the base standoff from the observed opening (0.35–1.5 m).
`lateral_offset` selects a shift along the opening (−0.6–0.6 m; positive is
robot-right when facing the cabinet). These are requested relative dock
parameters, not simulator coordinates. Omit them for automatic candidates.
Native route/arm planning and physical collisions remain active; Qwen controls
when to reposition, not whether an impossible trajectory becomes executable.
The Qwen cabinet profile requests `--local-paths-only`: it tries checked local
IK/joint paths and returns failure without the slow global cspace fallback for
each candidate. Qwen may then change its dock. This is a search tradeoff, not
a claim that all reachable paths will be found; standalone placement keeps its
existing global fallback by default.

For `place(..., redock=True)` and other optional-redock skills, an unsuccessful
redock no longer vetoes the requested manipulation. The skill attempts planning
from the current live pose and the response retains the failed `redock` result
separately. Invalid arguments still report errors before motion. Placement
success is never inferred from docking success. Tests cover both returned
redock failure and a runtime exception followed by an actual skill attempt.

This is local visible-target navigation. It does **not** register another
cross-room planner or claim to implement exploration. Use the existing sensor
navigation backend through `QwenSkillInterface.from_navigation` for that scope.

The drawer/cabinet must be opened before picking an object that will occupy the
only gripper. Opening routines include the necessary contact motion after a
handle grasp; they do not ask Qwen to infer a door hinge's ground-truth pose.
Ordinary loaded navigation never opens the gripper to make the route easier.

### Accessible-opening approaches for pick/place

The Qwen facade accepts `approach_mode="auto" | "top_down" | "front_entry"`
within the existing `pick` and `place` skills. These options do not modify the
native oracle catalog or the qualified set-table dispatch. Exposed supports and
open drawers use a vertical approach; an overhead-obstructed shelf uses an
outside-alignment and horizontal-entry sequence. The distinction is the observed
opening, not a receptacle-name rule.

Auto uses a remembered observed front opening when available; otherwise it
estimates overhead coverage over the reconstructed support. This is a heuristic,
not proof of clearance or a tested general room classifier. Qwen can override
it from the camera images. All pose segments still require native arm/payload
planning and physical execution. Missing geometry is never replaced by fixture
truth.

```python
place(receptacle="kitchen drawer", region="interior", approach_mode="top_down")
place(receptacle="kitchen storage cabinet", region="interior",
      approach_mode="front_entry", entry_clearance=0.12, entry_height=0.04)
pick(target="cup", source_receptacle="fridge", approach_mode="front_entry")
```

`entry_clearance` (0.03–0.30 m) reserves space for the entire observed object and
calibrated gripper outside the front plane. `entry_height` (0–0.15 m) adjusts
vertical insertion clearance. `wrist_yaw_degrees` (−90–90) ranks preferred
placement orientations; pickup preserves learned grasp contact orientations.
Front pickup withdraws through the opening before lifting higher. Front placement
uses sensor-derived upright object proposals rather than rotating a learned
contact without checking its support. These are geometric refinements, not
learned M2T2 predictions. Parameters apply to the current call; no scene reset.

Implementation does not establish success for every container. See component
results for actual trials; fridge/microwave insertion needs its own qualification.

### Drawer withdrawal and subsequent motion

The Qwen drawer profile resolves arm covering-sphere/world overlap against the
calibrated arm collision-shape boxes. The same 5 mm clearance margin, native
self-collision check and physical collisions remain enabled. It tries a checked
outward withdrawal, then checked upward/outward alternatives if necessary.
An unavailable path still reports failure.

After successful release, withdrawal and endpoint settling, this profile retains
that arm pose instead of unconditionally tucking. This does not authorize the
next motion: `SensorRedocking` checks its entire base route with the retained arm
shape, and manipulation plans from measured joints. The existing non-Qwen
fixture posture behavior and qualified set-table pipeline remain unchanged.

The refrigerator contact profile currently targets the visible vertical door
pull rather than the lower horizontal freezer-drawer pull. It uses visual
pointing and depth, not fixture IDs. Other refrigerator handle arrangements
are not qualified by this profile.

## Observation and state continuity

- Execution receives onboard images, RGB-D geometry, robot calibration,
  proprioception, and estimated SLAM state. No target IDs, fixture joint values,
  reference trajectories, or simulator object poses are execution inputs.
- A successful pickup supplies the measured payload geometry for subsequent
  carrying/docking. A component reset that starts with an object held reconstructs
  it from current depth. Measured finger geometry is used for extraction: the
  full-stroke gripper planning envelope must not erase the held object.
- Initial loaded mapping excludes that measured payload from static occupancy.
  The same payload remains in the collision envelope during motion.
- Redocking invalidates old grasps and placement plans. An explicit successful
  placement redock can be consumed by the following place call without repeating
  perception/docking. Other motion invalidates it.
- Fixture handle observations are passed from redocking to grasping. Observed
  cabinet aperture memory persists after opening and is reanchored with SLAM.
- Failed calls leave the current physical state intact. There is no rollback.
  Full diagnostics remain on disk; model feedback is bounded and reports the
  failed stage and an allowlisted sensor reason: hidden support, unreachable
  placement, or unavailable loaded dock. A failed dock may still have improved
  the wrist view; Qwen can then try current-pose placement with full arm/payload
  planning. Tool-list results are delivered even when code omits `print()`.

The stove uses the explicitly authorized
[seven-knob appliance manual](appliance_control_conventions.md). Qwen identifies
the pot's burner from the image. The backend selects its knob by counting the
complete observed row. This is supplied appliance knowledge, not recognition of
an invisible association. Incomplete control-panel views can still fail.

## Component trials

```bash
cd /nobackup/le/cross-episode-qwen-capx
export CUDA_VISIBLE_DEVICES=5
export SENSOR_MODEL_ENDPOINTS="$(cat /path/to/running/services/endpoints.json)"
bash scripts/test_qwen_household.sh \
  --case open_cabinet \
  --recording /path/to/pregrasp/state_recording \
  --output runs/new_unique_trial --render
```

Cases: `place`, `open_fridge`, `open_cabinet`, `open_drawer`, `cup_in_drawer`,
`bowl_in_cabinet`, `pot_on_stove`, `pot_knob`. Use `--stove-convention` explicitly
for the compatible seven-knob appliance. `--frame-index` selects a private reset
frame. `--floor-context` restores only earlier measured floor/visited-footprint
evidence in that same recording. `--observed-aperture` is only for a previously
measured aperture expressed in the reset's base frame, not authored geometry.
The Qwen runner uses `--max-place-candidates 48` per attempt so the model can
redock instead of exhausting hundreds of candidates. Set zero for exhaustive
search. Standalone backend defaults remain zero; this does not change the
qualified pipeline's candidate policy. This is a search budget, not a motion
compression-factor cap.

`scripts/test_qwen_household_suite.py --cases cases.json --output runs/suite`
runs independent cases sequentially, keeps failures, and writes one live log
and video per case. Example case entry:

```json
{"case":"open_drawer","recording":"/path/to/state_recording","frame_index":0}
```

Outputs include `transcript.json`, `tool_calls.json`, `sensor_skill_calls.json`,
backend logs, `report.json`, actual-state recording, and deferred `demo.mp4`.
Rendering occurs after control stops, with navigation RGB/depth above wrist
RGB/depth. There is no video speed-up or simulator reset between calls.

## Qualification limits

### Interactive fridge panel control

The Qwen household profile now yields after the handle pull and checked reach to
the rear panel. `open_door` returns `partial_open_panel_contact_ready`; this is
**not** a successful full opening. Fresh RGB views and the `door_state` field
let Qwen choose the next movement instead of restarting handle acquisition.

```python
state = door_state(target="fridge")
result = continue_door(target="fridge", motion="base_push",
                       advance_degrees=15, wrist_rotation=0.5)
# The equivalent execute_skill("continue_door", {...}) spelling is accepted.
```

- `arm_swing`: follow the observed hinge arc with the arm.
- `base_push`: coordinate arm motion and planar base translation.
- `base_rotate`: also rotate the base along the planned contact arc.
- `advance_degrees`: 1–90 degrees of additional observed opening requested per
  call, limited by the overall opening target. Intermediate calls retain the
  closed pushing gripper; full opening includes release and withdrawal.
- `wrist_rotation`: 0–1, the fraction of the door rotation assigned to the wrist.

These are continuation helpers for the same door skill, not new oracle atomic
skills. The scene stays live. In-memory context contains observed panel points,
estimated hinge/SLAM geometry and robot calibration; no simulator fixture pose,
private score or trajectory checkpoint is passed to Qwen. Other robot motions
invalidate the local continuation context. All calls retain native collision
planning and physical contact. `contact_pose_reached` is an executed pose, **not
a measured contact-force claim**. `opening_completed` reports command completion;
the independent physical scorer still decides task success.

The near wrist view now refits a newly visible rear liner before pushing, and
alternate contact IK branches plus an explicit free-edge waypoint avoid treating
a reachable endpoint as a reachable path. Broad robot covering-sphere overlaps
are checked against calibrated collision-shape boxes at the existing margin.
The fixed-command diagnostic (`--policy skill`) retains automatic full opening;
Qwen trials enable the interactive handoff. Physical qualification is tracked in
[results document](qwen_household_results.md). `qwen_household_fridge21` passed
full opening, release and withdrawal under Qwen control. This is one component
trial, not broad robustness qualification. A separate withdraw–redock–recontact
fallback is not implemented; the passing run used continued base rotation.

These are development component trials, **not** a held-out success-rate benchmark
or eight successful full composites. Placement trials start with an object
already held; fixture-opening trials start before grasping. The cup/drawer
reset uses recorded qpos/commands with zero initial velocities and is identified
in its provenance file. The execution code receives neither that source path's
actions nor reference targets.

Final physical scoring is private. It checks the initially held object against
the semantic support and released-hand condition; stove placement checks the
requested burner. Knob scoring independently associates the actual pot/burner
and rejects rotation of unrelated movable controls. Opening requires at least
80% of the physical range and a released gripper. A command-completed response does not establish release or semantic success.
The tool explicitly reports when visual settling was inconclusive; Qwen must
inspect fresh images. Command completion, Qwen's
`finish()`, and physical success are reported separately.

Control commands run at 50 Hz. The shared motion requirements in
[human_like_motion.md](human_like_motion.md#mandatory-defaults) still apply.
Several contact/docking components exceed 15 simulated seconds; this extension
is **not timing-qualified across all skills**. Reports retain measured durations
and the over-target status. Faster videos or successful flags do not establish
that redundant posture changes or contact motion are optimized.

## Measured development results

See [the result table](qwen_household_results.md) for run-specific outcomes,
durations, and videos. Failed trials are retained for diagnosis.

`align(receptacle, entry_clearance=.10, entry_height=.04,
wrist_yaw_degrees=0.)` is a placement-phase helper exposed to Qwen. It plans and
executes only the front-opening alignment, keeps the grasp, and returns for a
fresh observation before insertion. It consumes its prepared plan even on error;
subsequent placement must plan from the current physical pose. It neither adds an
atomic task category nor treats alignment as completed placement.

For a recorded-start diagnostic use `--policy skill --case bowl_in_cabinet
--component-align-only` with `scripts/test_qwen_household.sh`. The report separates
`alignment_success` from the unchanged private task `success` score.

For placement, `front_wrist="object_level"` retains the observed object-plane
orientation proposal; `front_wrist="horizontal"` explicitly points the gripper
into the opening, with checked roll candidates. This latter option can tip an
attached object: drawer-handle orientation is not automatically appropriate for
a held bowl. The cabinet horizontal-wrist trial executed but failed physical
placement. Neither option is a guarantee that a partially seen bowl is level.
`align(...)` accepts the same option for inspection without release.

After front placement releases, the gripper exits horizontally at release height,
with unchanged wrist orientation. Only outside the observed opening does it
restore the original alignment joint branch and reverse the free approach to the
pre-placement posture. Keep the gripper open throughout. Do not reverse the final
descent first: that lifts the fingers under an object that may still be touching
them. No extra tuck is imposed. Open-top placement behavior is unchanged.

This path uses the observed aperture and calibrated gripper envelope, not hidden
simulator contact flags. An outside-opening waypoint does **not** prove that an
object has separated from the fingers: an early cabinet trial still carried the bowl
on a pad when the shelf had not supported it at release. A subsequent deeper
placement passed (`approach_modes_cabinet09`); this is one recorded-start
component pass, not cross-scene qualification. Horizontal-wrist proposals also keep
the estimated object footprint behind the observed front plane, with clearance;
this is a planning constraint, not a relaxed scoring region.

### Qwen docking selection and unknown floor — November 10, 2026

`test_qwen_household.py --qwen-docking` enables an experimental docking
subpolicy. It broadens local empty-hand candidates across standoffs and
bearings, and ranks diverse candidates using Qwen3-VL with onboard images and
a numbered observed-map image. Qwen placement uses sensed navigation anchors;
it does not require a placement footprint or arm IK before docking. Qwen returns candidate IDs only. The first selected goal
gets a checked direct route or local A*; an explicitly logged multi-goal
fallback is allowed if that route is unavailable. Qwen selection does not
prove grasp/placement feasibility: the arm planner and physical evaluation
still run. Candidate maps and endpoint rejection evidence are saved even when
base movement cannot begin.

**User decision:** black/unknown floor alone must not veto local redocking.
`SensorRedocking` therefore sets `require_observed_support=False` for its local
swept checker. It still checks all observed obstacle points against the full
retained robot shape, keeps physical collisions enabled, and executes through
SLAM feedback and the native base controller. Unknown cells remain unknown;
they are not relabeled free. This is a local approach policy, not proof of
collision-free unobserved space, and does not alter cross-room exploration or
the accepted loaded table-placement backend. Physics trials are required
before claiming fridge pickup success.

The user subsequently requested returning fridge pickup to the previous
**automatic** docking choice. Use the component harness without
`--qwen-docking` for that comparison; the Qwen ranking experiment remains
opt-in. This does **not** revert the unknown-floor traversal decision above.
The accepted Qwen-ranked table example remains saved and its behavior is not
changed by switching the fridge test's mode.

### Basket docking default — November 11, 2026

The grocery recipe's local basket-placement dock now calls the same
`rank_docks()` / `plan_ranked_docks()` Qwen subpolicy. It previously bypassed
Qwen and called the automatic multi-goal planner directly. This default applies
to `scripts/test_basket_place_redock.py`, including calls through
`runtime/basket_placement.py`; no extra launch flag is required.

The sequence is: visible receptacle RGB-D point → collision-checked facing
candidates → Qwen ranks numbered poses → checked direct route or A* → physical
arrival → fresh close-range RGB-D → grasp/placement planning. Candidate
standoffs span 0.50–0.75 m. The first dock must not depend on placement
feasibility, a complete basket floor, a supported payload footprint, or a
hypothetical arm IK solution. This supersedes the earlier pre-dock placement
probe and witness requirement. `BasketDockProbe` remains only as a historical
diagnostic utility; production docking does not call it.

`observe_receptacle_approach.py` supplies only a semantic navigation anchor.
Its `approach_anchor_only` evidence cannot be used as a release surface. Distant
basket points are no longer exported as post-dock floor memory. Placement uses
fresh observations and retains physical support/containment scoring. If actual
close-range manipulation fails, Qwen can redock and observe again; docking success
alone is never reported as manipulation success.

Basket support extraction measures depth normals before selecting a horizontal
surface. Within the visually identified, bounded basket mask, it selects the
lowest observed horizontal parts and merges coplanar patches. This avoids
mistaking a height-sliced wall band or the rim for the floor when the VLM point
lands near the edge. It uses no mesh dimensions or private basket pose.

The basket skill first points at the receptacle from the carry endpoint. If
no visible RGB-D target can be located, it acquires a closer view before raising
the payload. It does not test a release footprint at this stage. Straight lattice segments and consecutive turns are merged only
when their swept checks pass. This view-acquisition motion is distinct from the
Qwen-selected manipulation dock. A failed observation is never passed downstream
as a nonexistent input cloud. Elongated payloads receive upright-first placement
proposals; basket release starts at 5 cm clearance. A failed placement keeps the
held-object handoff rather than running an empty-hand return.

These changes are undergoing complete-episode qualification; see
[qwen_household_results.md](qwen_household_results.md) for physical component
results and limitations. Unknown floor alone does not veto this local redock;
observed obstacles and the held payload remain in swept checks. The existing
Qwen planner explicitly records geometric fallback if its first choice has no
route. Reports include `docking_policy` and `qwen_selections`, so a Qwen call,
a fallback and a failure before selection can be distinguished.

Cross-room routing and the qualified set-table placement are unchanged. This
integration alone does not establish grocery task success. The recorded-state
regression is `scripts/test_recorded_basket_retry.py`; it restores the physical
state and previously observed floor, and rebuilds local obstacles from RGB-D
because the original full 3D map handoff was not exported. Full persistent-map
qualification still requires a complete episode.

### Shared horizontal entry for confined pick/place — November 10

Confined pickup now uses `horizontal_entry_rotation`, the same tool-frame
construction used by horizontal cabinet placement: the tool approaches into
the observed opening and its closing line is level. `front_pick_candidates`
retains sensed M2T2 contact centers but explicitly adapts their orientations;
these are documented geometric adaptations, not untouched learned proposals.
The shared `plan_front_alignment` / `plan_front_entry` then aligns outside and
inserts horizontally. Pickup enters at contact height (`entry_height=0`);
its support-breaking lift is planned after finger closure. Placement retains
its existing release-height behavior. The observed collision world, calibrated
robot shapes, physical contacts and physical goal score remain active.

Malformed grasp transforms are discarded before rotation/IK operations instead
of aborting the rest of the candidate batch. The report's `front_horizontal`
now reflects the pickup backend's executed/requested horizontal mode when a
backend report exists, rather than only the cabinet diagnostic CLI flag.
An alignment-only pickup diagnostic leaves the gripper open and reports
`alignment_success` separately from private pickup success.

Qwen docking remains opt-in outside the basket recipe. Pickup docking no longer
requires horizontal-grasp IK: top-down versus front-entry grasp orientation is
resolved after arrival and fresh observation. The arm stage still plans its sweep.
Burner docking follows the same approach-first rule. The non-Qwen qualified
set-table path keeps its original dispatch.


Confined `front_entry` pickup now executes a checked open-gripper alignment and
refreshes onboard RGB-D/M2T2 proposals before grasping. The skill remains one
public `pick` call. The alignment diagnostic flag stops after alignment; it
never counts as pickup success. Top-down pickup and table placement do not use
this added cycle. An empty M2T2 result is reported as `no_grasp_proposals`, not
as proof that the target is invisible.

`observed_support_below_target` handles multi-shelf depth views for explicit
front entry, without changing the legacy top-down support selection.
`support_break_clearance` permits only initial support contact during a vertical
lift, checks other obstacles throughout, and requires complete clearance before
horizontal withdrawal. The ordinary physical grasp/hold score is retained.
`plan_front_entry` can seed alignment from collision-checked contact IK branches;
placement retains its existing path. These changes have focused regression
tests, including rejecting a wall collision during support lift-off.

### Basket and stove placement through the interface — November 10

Use the existing `place` skill, with a semantic receptacle:

```python
execute_skill('place', {'receptacle': 'purple basket',
                       'region': 'interior', 'approach_mode': 'top_down'})
execute_skill('place', {'receptacle': 'front left stove burner'})
```

Basket interiors dispatch to the container backend. Before perception, a
collision-checked retained-grasp posture exposes the bottom; the usual generic
table backend is unchanged. Current-pose placement does not require a fictitious
successful base dock. Explicit `redock` remains available separately; these
placement component trials do not qualify far-start navigation.

Stove placement retains the native observed-plane settling proposals instead
of overriding all of them with forced upright poses. The target is pointed in
fresh onboard images, with front/rear relative to the control panel/backsplash.
Neither these geometric proposal adaptations nor the basket fallback should be
described as untouched learned M2T2 outputs. cuRobo/observed-geometry path checks,
physical gripper release, and private semantic scoring still apply.

The live `HouseholdSession.execute()` preserves pre-grasp observed payload
geometry using `observed_pick_payload`. A mid-episode component reset must also
restore that observation: `test_qwen_household.sh --held-observation INPUT_NPZ`
accepts the same episode's sensor `held_points` and calibrated `ee_pose`. It
converts those points to the tool frame; it does not load object truth, a goal
pose, or actions. Without this context, reconstruction behind the closed fingers
may cover only part of a vessel. The option is a component-fixture input, not
extra information provided to a policy in normal live execution.

See `qwen_household_results.md` for measured physical outcomes and limitations.

### Microwave grasp orientation choices — November 10, evening

**Default order: top-down first.** Qwen's first microwave pickup attempt must
use `approach_mode="top_down"`; it may inspect or redock beforehand. Front-entry
alternatives follow only after a failed or planner-infeasible top-down attempt.
The fixed-command microwave pickup test uses the same top-down default. This is
a planning preference, not evidence that the preferred grasp succeeds.

Qwen can explicitly compare these pickup calls:

```python
pick(target="food container", source_receptacle="microwave", approach_mode="top_down")
pick(target="food container", source_receptacle="microwave",
     approach_mode="front_entry", front_wrist="horizontal")
pick(target="food container", source_receptacle="microwave",
     approach_mode="front_entry", front_wrist="learned")
```

`top_down` approaches the learned contact from above. `front_entry` describes
how the robot enters the cavity, independently of wrist orientation. With
`learned`, it preserves M2T2's proposed orientations (including downward rim
grasps), rather than replacing them with horizontal poses. This is pick-only;
placement retains `object_level` and `horizontal`. Learned front entry plans
alignment, insertion and withdrawal before executing, without the separate
alignment/re-observation cycle used for horizontal bottle pickup. All paths
still undergo swept collision checks and physical grip verification.

A named `source_receptacle` selects the observed support *below* the target for
both orientations; the appliance roof must not become the support. Shallow confined
pickup uses 1 cm local obstacle voxels with the original padding/margins. For
shallow front-entry targets, calibrated palm boxes resolve conservative palm
sphere overlap; this does not disable physical or planned collision geometry.
These changes are not proof that either orientation succeeds in a microwave.
See the measured component results.

Microwave placement points to the visible interior turntable and extracts its
connected horizontal RGB-D surface. The enclosing appliance's instance mask
may have a hole at the cavity, so it is not used to exclude that floor. A clean
image plus a small point marker lets Qwen reject counter/door points without
repainting the turntable as a red object. The held surface and opening geometry
may be retained from the same episode's RGB-D/SLAM observations. A remembered
opening also supplies loaded approach docking; no simulator appliance pose is
used as an execution target.

### Microwave pickup calibration (November 10)

The microwave Qwen component test now enforces one action per model turn.
After a pick or redock, it supplies fresh navigation/wrist RGB and the action's
planning/contact feedback before another action. A cell that batches multiple
actions is rejected before motion and returned for correction. This is a model
interaction boundary, not an extra physical success criterion.

The first pickup is top-down. After an infeasible attempt, Qwen may request:

```python
pick(target="food container inside microwave", source_receptacle="microwave",
     approach_mode="front_entry", front_wrist="tilted_rim",
     wrist_tilt_degrees=40, rim_height_offset=-0.001,
     wrist_yaw_degrees=0)
```

`tilted_rim` keeps the proposed contact's horizontal position, sets its height
from the sensed upper rim plus `rim_height_offset` (-0.02 to +0.03 m), and tilts
from vertical downward toward the cavity (0–65 degrees). The wrist therefore
leans outward. Yaw rotates the closing direction. These are geometric grasp
hypotheses, not unchanged learned M2T2 predictions or oracle grasp annotations.
The existing IK, swept geometry, grip verification and withdrawal planner must
still qualify each hypothesis. This option is pick-only.

An empty-hand `redock` also accepts `lateral_offset` (-0.6 to +0.6 m; positive
left relative to target bearing) alongside standoff `distance` (0.35–1.5 m).
An explicit offset selects that candidate, with A* and retained-posture sweep
validation; omitting it preserves the existing automatic candidate selection.
Loaded cabinet/appliance docking retains the opening-relative convention; a remembered microwave/fridge opening now routes standalone redock through the aperture backend. Qwen local loaded docking allows unknown floor while retaining observed obstacle and payload sweeps; global navigation defaults are unchanged.
All calibration targets use onboard depth/SLAM, not simulator object poses.

For placement, `placement_preference="center"` chooses among already-supported,
unoccupied contacts near the sensor-derived center (4 cm candidate tolerance).
A semantic wrist floor point selects the floor height; the center comes from
the measured horizontal support extents in the observed opening frame, not
the pixel itself. This is an estimated cavity-floor center, not a fitted circle. A missing reachable centered candidate is reported
as failure; it does not silently place at the front edge. Other callers default
to the existing `nearest` preference. The microwave component command now asks
for centered placement; this is an explicit placement goal, not a new semantic
success constraint for every microwave task.

Horizontal placement also accepts `wrist_roll_degrees` (-180 to 180); omitting
it preserves the existing roll search. Centered placement seeds alignment from
up to three checked contact IK branches, retaining full insertion, payload and
withdrawal checks. A complete command sequence is not evidence of successful
release: the separate physical scorer remains authoritative.

Tilted placement accepts `object_tilt_degrees` (0–45) with
`front_wrist="object_level"`. This rotates the observed held object and attached
tool together about the observed opening's horizontal tangent; it is distinct
from pickup's `wrist_tilt_degrees`, measured from downward vertical. For example:

```python
execute_skill("place", {
    "receptacle": "microwave", "region": "interior",
    "approach_mode": "front_entry", "front_wrist": "object_level",
    "placement_preference": "center", "object_tilt_degrees": 40,
    "entry_height": 0,
})
```

For tilted shallow objects, the measured cloud's thin axis avoids mistaking a
dominant visible side wall for the bottom plane. Calibrated finger geometry is
checked closed for insertion and open for withdrawal. Several release heights
are tried per candidate. The successful component and timing exceptions are in
`qwen_household_results.md`; this is not full-task or pickup qualification.
Front-entry pickup defaults to zero extra entry height when none is supplied;
explicit caller values are retained. Required support-break lift and horizontal
withdrawal remain planned and physically verified.

### Tilted microwave place-and-pick calibration

For `front_wrist="tilted_rim"`, positive `entry_height` now defines a raised
pre-grasp **along the selected tool axis**. The robot enters at that height,
then descends along the tilted axis to the rim. Zero keeps the former level
entry. Placement's insertion/withdrawal is unchanged.

For shallow observed containers, the pickup backend also constructs side-rim
hypotheses from the upper RGB-D boundary. These keep the jaw closing line level
while tilting the approach toward the opening; they are explicitly labeled
`observed_side_rim`, not learned M2T2 predictions. M2T2 still supplies target
grasp proposals, and its calibrated variants remain in the search. Endpoint
linkage ranking avoids spending IK time on obviously blocked contacts; full
insertion, withdrawal, native collisions and physical hold verification remain.
`wrist_yaw_degrees` calibrates the learned-contact variants; side-rim orientation
comes from the observed opening/side geometry.

The qualified pickup component (`microwave_pick21_side_rim`) uses:

```python
execute_skill("pick", {
    "target": "food container inside microwave",
    "redock": True, "redock_distance": 0.8,
    "source_receptacle": "microwave",
    "approach_mode": "front_entry", "front_wrist": "tilted_rim",
    "wrist_tilt_degrees": 55, "rim_height_offset": 0.005,
    "entry_height": 0.04,
})
```

These are calibrated semantic-skill arguments, not a claim of autonomous Qwen
selection or universal microwave settings. For a continuous component cycle,
add `--pickup-after-place` to `scripts/test_qwen_household.sh --policy skill
--case place_microwave`. This diagnostic performs the pickup above after the
placement command in the same live simulation, with no intermediate state
restore. Private scoring requires released support followed by pickup of that
same object. It is not a full meal-preparation task.


### Basket frontal docking and post-arrival view recovery

Basket docks use the dominant observed vertical counter/front surface below the
RGB-D target to estimate the accessible facing direction. This uses only local
points and the robot's estimated pose, not fixture meshes or world axes. When
that estimate is available, docks lie on and face along its measured normal
without quantizing the heading; reports retain the fitted evidence and whether a frontal
estimate was available. Navigation still does not require placement IK.

If the first close-range placement observation cannot produce a usable plan
before any insertion/release motion, the basket skill can move the retained
object sideways using checked arm/payload paths, then capture fresh RGB-D.
The optional lateral clearance parameter leaves other fixture posture defaults
unchanged. Earlier failed attempts stay in the log, with `recovered_by` only if
a subsequent placement actually succeeds. This addresses the observed gripper
occlusion; it is not permission to fill unseen floor geometry.

The post-arrival view helper projects the sensed basket anchor into the onboard
navigation image. If robot-mask pixels obscure that region, it requests a
checked sideways clearance posture before close-range reconstruction. This
visibility measurement triggers a recovery motion, not a navigation veto.
Fresh support patches must remain associated with the approached basket; a
segmented patch over 0.45 m laterally from that sensed anchor is not used as its
release surface. This association is basket-specific and runs after docking.

The initial semantic point may lie on the near rim, approximately 20 cm ahead
of the actual interior. After approach and view clearance, the basket skill
now calls fresh `--observe-only` perception before arm planning. It computes
an anchor from the dominant observed support height and requests another Qwen-
ranked, A*-planned dock if that interior is not centered within 5 cm laterally
and 0.50–0.65 m ahead (also aligned to the measured front when available).
This second dock uses the actual close-range interior, not the original rim
point. Placement reacquires RGB-D after the base moves. The initial approach
still requires neither arm IK nor a complete placement footprint. This is
implemented in `runtime.basket_placement.refine_basket_dock`; physical results
are tracked separately in `qwen_household_results.md`.


### Shared full-task Qwen dispatch — November 11

`scripts/run_sensor_household.py` now constructs `QwenSkillInterface` for all
five task recipes. It wraps the existing full-task navigation/exploration
handlers; it does not substitute the component session's local-only navigation.
`runtime/qwen_pipeline.py` owns the adapter and fresh local redock handoff.
Pickup, door/drawer and knob/button calls default to local redocking. Qwen ranks
sensor-derived poses; A* checks and executes travel. This is Qwen-guided docking,
not a claim that a Qwen agent generates the full high-level recipe.
Placement keeps its loaded docking inside its backend, including the new basket
interior refinement. Explicit placement redock uses the household session's
prepared placement. Failed local docking stays in the public result; the
manipulation handler still attempts from the live pose. Redock wall time and
feedback are included in `sensor_skill_calls.json`.

Fresh target coordinates from redocking replace stale navigation targets before
pickup. The prior M2T2-only runner, including the qualified set-table behavior,
remains selectable with `--legacy-m2t2-interface`. The default Qwen path is new
and requires separate physical qualification for each task. See the per-run
`interface_dispatch.json` and live `SKILL_INTERFACE` / `QWEN_PIPELINE_REDOCK`
messages to verify actual dispatch.
