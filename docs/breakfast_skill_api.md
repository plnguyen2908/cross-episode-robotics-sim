# Breakfast and coffee skill API

This describes the existing executor used by coffee and breakfast-with-coffee.
`list_skills()` supplies descriptions and argument schemas to the symbolic planner;
Qwen receives this catalog in its prompt. These are general tool contracts, not a
scene's oracle plan or hidden state. Always use the live catalog's enum values.
Standalone task interfaces may expose different skills.

Calls have the form `{"skill": "name", "arguments": {...}}`. All arguments are
strings. No-argument skills require `{}`. Inspect success/error feedback after
every call: issuing a call does not mean it executed successfully. A rejected
precondition produces no motion. Repeating it without changing its prerequisites
will not resolve the failure.

## Selecting a vessel

- The public interface accepts asset types. Pickup privately selects a matching
  object visible in the current input cameras, independently of previous navigation.
  Visibility is checked again on every pickup using occlusion-aware simulator
  rendering. Instance IDs and segmentation are not sent to the planner.
  Once grasped, that object's identity is preserved for transport and placement.
- The grounds box is `CoffeeDosingBox` and uses dedicated no-argument skills.
  It is not selected through the cup/bowl arguments.

## Navigation: argument combinations

`navigate(asset_type="Mug")` finds and approaches a mug. You do not need to
supply `destination` for finding an object: omission defaults to `source`.
Explicit transport destinations retain the meanings below.

`navigate` moves and docks the base. It never implicitly grasps or releases.
The selected vessel and the destination have separate meanings. This API is
vessel transport, not unrestricted navigation to any named room or object.

| destination | Supported vessel | Meaning and requirement |
| --- | --- | --- |
| `source` | Cup or bowl | Approach the object. May carry another object while approaching. |
| `spout` | Cup | Carry that held cup to the machine. Also permits returning empty-handed to that same cup already ready under the spout. |
| `home` | Cup | Carry that held cup to its recorded home counter spot. Its home must be known. |
| `serving` | Cup or bowl | Carry that held vessel to its dining place setting. |

Holding a grounds box does not satisfy a cup's transport precondition. Holding
another object is allowed during `source` navigation.

Optional `dock` is an approach candidate index string (default `"0"`); it changes
the approach, not the selected vessel. Use only indices listed in the schema.
Optional `setting`, when exposed, is `"default"` or a numbered dining setting.
An explicit number is valid only with `destination="serving"` and must exist in
this scene. `default` uses the vessel's configured serving position. Subsequent
`place` uses the position chosen during navigation; it has no `setting` argument.

## Pickup and placement

| Call arguments (plus vessel selector) | Requirement | Successful effect |
| --- | --- | --- |
| `pick(source="surface")` | Empty hand; matching object visible in current camera views | Lift selected cup/bowl; leave it held. |
| `pick(source="spout")` | Empty hand; matching visible cup physically under spout | Extract and hold cup. |
| `place(destination="spout")` | Selected cup held | Release cup under machine. Does not brew. |
| `place(destination="home")` | Selected cup held | Release cup at its home counter. |
| `place(destination="serving")` | Selected cup/bowl held | Release vessel at dining setting. |

Pick and place do not navigate the base. Bowls support surface pickup and serving
placement only. No previous navigation call is required. Visibility does not guarantee reachability: physical grasp or placement attempts may fail. Pickup may select a different visible type from the previous navigation target.

## No-argument skills

| Function | Prerequisites | Successful effect |
| --- | --- | --- |
| `detach_portafilter` | Empty hand; portafilter seated/locked | Detach and support it on loading counter; hand empty. |
| `pick_dosing_box` | Empty hand; box visible | Hold grounds box. No grounds poured yet. |
| `pour_grounds` | Grounds box held | Pour into detached portafilter, return box upright; box remains held. |
| `place_dosing_box` | Grounds box held | Park and release box; hand empty. Does not pour. |
| `attach_portafilter` | Portafilter physically detached (empty or filled) | Insert and lock portafilter, release; hand empty. |
| `press_power` | Empty hand | Press power button. Warmup is handled separately. |
| `wait_until_ready` | Machine powered; occupied hand allowed | Wait for machine warmup. Does not start brewing. |
| `press_brew` | Empty hand | Physically press brew. The machine accepts a cycle only if its physical operating conditions hold. |
| `wait_until_brewed` | Brew started; occupied hand allowed | Wait for completed brew with cup under spout. Cup remains there. |
| `wait_for_bowls` | Requested bowls in kitchen; occupied hand allowed | Wait for authored human filling event. Does not pick or serve. Only exposed in full breakfast. |

## Local usage examples

Approach and pick an available mug (the standard interface):

```json
{"skill":"navigate","arguments":{"asset_type":"Mug"}}
{"skill":"pick","arguments":{"asset_type":"Mug","source":"surface"}}
```

Carry that held mug to the machine and release it:

```json
{"skill":"navigate","arguments":{"asset_type":"Mug","destination":"spout"}}
{"skill":"place","arguments":{"asset_type":"Mug","destination":"spout"}}
```

Carry a held bowl to an existing dining setting and release it:

```json
{"skill":"navigate","arguments":{"asset_type":"Bowl","destination":"serving","setting":"1"}}
{"skill":"place","arguments":{"asset_type":"Bowl","destination":"serving"}}
```

After successfully picking the grounds box, pour and free the hand:

```json
{"skill":"pour_grounds","arguments":{}}
{"skill":"place_dosing_box","arguments":{}}
```

Internal datagen plans retain private roles; they are not planner-facing arguments.
These examples assume their prerequisites, and are not a complete episode plan.
Existing evaluation processes retain their loaded catalog; launch a new evaluation
to use revised descriptions. Dataset regeneration and model-server restart are
not required for documentation updates.

## Backend contract

The same type-level request can condition an oracle skill or a VLA skill. The
backend must preserve the selected instance across approach, pickup and placement,
and use the actual held object for transport. A VLA must implement selection and
tracking from its supported observations; it does not inherit the oracle's private
instance resolver. Coffee/current physical VLA execution remains unsupported until
an independent action adapter and physical success checks are available.

Dataset evaluation uses this interface without a version flag. Historical logs
retain their original schema and should not be interpreted as type-level runs.

Navigation and pickup also accept other scene object types, such as `Plate`,
`Apple`, `Vase`, `SaltShaker`, and `OilBottle`. The live catalog lists types from
this scene, without instance IDs, counts or positions. Type names are not restricted
to an enum for these two skills. Missing types return `target_not_found`.

```json
{"skill":"navigate","arguments":{"asset_type":"Plate"}}
{"skill":"pick","arguments":{"asset_type":"Plate","source":"surface"}}
```

A previous approach does not restrict pickup. Any currently visible matching
instance may be attempted with an empty hand. An off-camera target returns
`target_not_visible`; pickup does not move the base. Generic pickup uses the existing annotated grasp planner and physical
hold checks. Missing annotations, unreachable grasps and failed physical attempts
return failure. Floor objects are approached near their actual position; floor
pickup remains subject to reach and collision constraints. Generic object placement
and transport destinations are not added by this change; those still use the
vessel-specific contract. Specialized coffee skills remain available.

To test navigation and pickup without Qwen (use an idle simulator GPU):

```bash
ROBOCASA_DIR="$PWD/external/robocasa/robocasa" /home/leos/.venvs/sim/bin/python -m scripts.test_any_vessel \
  --example datasets/test_set_v1/examples/example_0001 --asset-type Plate --gpu 7
```

Replace `Plate` with `Mug` to check the existing vessel path. Add `--render-video`
for a diagnostic video. The script prints a unique output directory containing
`result.json` and exits nonzero on a failed attempt. This is a two-skill test, not
full-task validation. No dataset regeneration is needed.

New evaluation sessions keep scattered objects physically movable; earlier sessions
froze them. Initial poses and goal membership stay the same, but this is a physics
configuration change recorded as `dynamic_clutter` in evaluation results. Datagen's
default is unchanged. Start a new evaluation process to load the expanded catalog
and dynamic-object configuration; the Qwen server does not need restarting.

## Coffee-machine navigation terminology

There is currently no registered `CoffeeMachine` target type. Calling
`navigate(asset_type="CoffeeMachine")` returns `target_not_found` if no such target
is registered; accepting a string in the schema does not imply that target exists.

The observed Qwen call in `runs/qwen_oracle_evaluation_ygz3w4de` was:

```json
{"skill":"navigate","arguments":{"asset_type":"MoonlakePortafilter","destination":"source"}}
```

`MoonlakePortafilter` is the portafilter asset type listed in the catalog. This
approaches that object at the machine; it does not select a coffee-machine asset.
`destination="source"` can be omitted for this approach. The controller may log
“navigate to coffee station,” but this is a motion caption, not the model's target
argument. Use the `planner_decision` and `oracle_skill` events to identify what the
model actually requested.

Separately, `navigate(asset_type="Mug", destination="spout")` transports the
selected held mug to the machine's spout (or returns empty-handed to its ready
cup under the documented exception). Navigation does not detach the portafilter;
that operation remains the separate `detach_portafilter` skill.

## Dependency semantics and verification

Attachment checks that the portafilter is physically detached; it does not check
grounds or require an empty hand at the symbolic boundary. An occupied hand can
still make the physical grasp fail. Button-press success means contact was made,
not that coffee was brewed. Final task success still requires the actual goal.

Regression coverage: `tests/test_skill_interface.py`,
`tests/test_any_vessel.py`, and `tests/test_visibility.py`.
These check tool admission and visibility; they are not claims that every visible
object is physically graspable or that every action ordering completes the task.

Run the physical dependency checks on a frozen example (choose idle GPUs):

```bash
ROBOCASA_DIR="$PWD/external/robocasa/robocasa" python -m scripts.test_skill_dependencies --case box --gpu 7
ROBOCASA_DIR="$PWD/external/robocasa/robocasa" python -m scripts.test_skill_dependencies --case attach --gpu 7
```

The box case powers on, approaches and picks the box before detachment, waits
while holding it, then approaches a mug while still carrying the box.
The attachment case detaches and reattaches without adding grounds.
Each run uses a unique `runs/skill_dependencies_*` folder and writes per-call
results to `result.json`. A physical failure is reported as failure rather than
treated as proof that the whole sequence works.
