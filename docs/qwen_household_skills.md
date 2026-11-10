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
`redock_target`, and optionally `redock_distance`. The loaded placement backend
chooses standoff using payload/support geometry and rejects a fixed distance
override rather than silently ignoring it. Explicit `redock()` uses `distance`,
not `redock_distance` (the latter belongs to `execute_skill` arguments).

```python
execute_skill("pick", {"target": "cup", "redock": True})
navigate(target="cup", destination="kitchen drawer")
execute_skill("place", {
    "receptacle": "kitchen drawer", "region": "interior", "redock": True,
})
```

This is local visible-target navigation. It does **not** register another
cross-room planner or claim to implement exploration. Use the existing sensor
navigation backend through `QwenSkillInterface.from_navigation` for that scope.

The drawer/cabinet must be opened before picking an object that will occupy the
only gripper. Opening routines include the necessary contact motion after a
handle grasp; they do not ask Qwen to infer a door hinge's ground-truth pose.
Ordinary loaded navigation never opens the gripper to make the route easier.

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
The Qwen runner uses `--max-place-candidates 24` per attempt so the model can
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
