# Room doors

An optional [latched room-door prototype](latched_room_doors.md) now supports
a physical rotating handle and catch. Its mechanism test passes; robot skill
validation is pending. The default doors described below remain unlatched.

New house builds fit physical double doors to each paired room doorframe.
New RoboCasa exports shift the living room north, retaining its dining-room
connection and adding a kitchen connection. Furniture moves with the living
room before task objects are populated. The living room is enlarged by 2 m
eastward and at least 2 m southward (more for deeper kitchens, preserving the
dining doorway); the sofa and coffee table move away from the shared
wall, preserving their physical sizes. Both kitchen wall layers are cut.
The resulting three-room scene has three connections:

| ID | Connection | Default |
| --- | --- | --- |
| room_door_0 | kitchen ↔ dining | closed |
| room_door_1 | dining ↔ living | open |
| room_door_2 | kitchen ↔ living | open |

Both leaves have visible handles, collision geometry, mass, and damped/frictional
hinges. An open door remains a physical object. Initialization selects hinge
angles; execution does not hide leaves or disable their collisions.

To override initial states, put `room_doors.json` beside the input scene XML:

```json
{
  "room_door_0": "open",
  "room_door_1": "open",
  "room_door_2": "open"
}
```

Unknown IDs or states fail setup. House metadata records the initial states,
joint names, and frame names. Room-door joints remain dynamic when other
non-task fixtures are frozen. Reachability caches distinguish door angles and
in-memory navigation masks are invalidated when those angles change.
Disconnected rooms remain represented in the physics map.

Composite scene configuration also accepts `scene.room_doors`, for example
`{"room_door_0": "closed"}`; unspecified doors remain open. This replaces
the previous hard-coded all-open composite setup. Recompute `scene_spec_sha256`
with `generation.composite_pilot.scene_hash` after editing scene parameters.

For the four cross-room composite tasks, set top-level `room_route` to
`open_shortcut` or `living_room_detour`. Both require the identical scene:
door 0 closed and doors 1/2 open. The shortcut recipe prefixes explicit
`navigate` and `open_door` calls for each direct-door leaf. The detour leaves
the door closed and relies on collision-aware navigation through the living
room. Neither adds a closing requirement. The kitchen-only warm-two-pots
recipe rejects this route option because it currently has no cross-room trip.

These configuration and plan contracts are unit-tested. The physical opening
test now passes for both non-latched leaves from the kitchen side: approximately
89.3 degrees after release, using three short strokes per leaf with local
repositioning. See the [robot video](../runs/qualify_room_door_v_n9a0f2/composite.mp4)
and [test report](../runs/qualify_room_door_v_n9a0f2/report.json). These local
artifacts validate opening, not closing or a complete composite route comparison.
Run the focused check with `python scripts/qualify_composite_fixture.py room_door`.

Preview the two plans before physical qualification:

```bash
python -m cross_episode_sim.generation.composite_pilot --describe \
  --task set_table --variant original --room-route open_shortcut
python -m cross_episode_sim.generation.composite_pilot --describe \
  --task set_table --variant original --room-route living_room_detour
```

Removing `--describe` starts a physical attempt, not a guaranteed-success demo.
The runner writes separate episode names for each strategy and recomputes the
scene hash. Both strategies use identical initial door states and task objects.

This changes new house builds; existing recordings are unchanged. A task goal
should require closing a door only when the instruction requests it.

## Validation and remaining work

Unit tests cover open/closed geometry, a blocked/clear crossing ray, physical
motion under applied torque, argument validation, and inclusion in navigation
collision checks. The real scene is also compiled and checked for initial
door/frame penetration.

These are scene and collision checks, not successful robot handle-manipulation
rollouts. Non-latched leaves now have explicit bindings to the shared
`open_door`/`close_door` backend, with separate fixed handle bodies so panel
contact is not mistaken for handle contact. Opening now has the physical
evidence linked above; closing and complete route alternatives remain pending.

The new connection graph is a triangle: the direct kitchen–dining door can
remain closed while the kitchen–living–dining route uses open doors. Existing
scene files retain their old layout unless explicitly regenerated. Geometry
and crossing-ray checks do not replace a full robot-navigation rollout; the
two alternative strategies still need end-to-end physical validation.

## Living-room furniture and evaluation variation

New exports include two armchairs, a bookshelf, a media console, and an end
table, alongside the sofa, coffee table, and side table. Training defaults are
fixed. Furniture uses collidable geometry; none of these are new public skills.

For later evaluation, `furnish_living_room(scene, transforms)` accepts a map
of stable group names to `dx`, `dy` (metres), and `yaw_deg` (degrees). These are
offsets from the training layout, applied before physics, object population,
or episode recording. Connected parts rotate about the furniture anchor as
one unit. Object dimensions and IDs are retained.

Example evaluation transform JSON:

```json
{
  "coffee_table": {"dx": 0.1, "dy": 0.1, "yaw_deg": 90},
  "armchair_left": {"dx": 0.2, "dy": 0.0, "yaw_deg": -30}
}
```

Pass its path as `--furniture-config` to `scripts/render_room_layout.py`, or
pass the dictionary as `furniture_transforms` to `export_initial_scene`.
Additional group names: `sofa`, `side_table`, `bookshelf`, `console`,
`armchair_right`, `end_table`. Apply to a fresh scene, not an existing furnished
recording. `living_furniture.json` records final poses, grouped bodies, and
conservative footprints for reproducibility. Invalid configurations are
rejected before writing the scene.

Preflight rejects footprint overlap, out-of-room placements, and furniture
entering the reserved 1.5 m shared-wall corridor. It does not establish robot
reachability, graspability, or success of a complete task. Those require rollout
validation before admitting an evaluation example.
