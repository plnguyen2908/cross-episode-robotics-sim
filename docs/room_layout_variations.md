# Configurable room extensions

The sampler now supports an optional `room_layout` group alongside the existing
procedure, object, furniture, door/route and appearance groups. The baseline is
the existing kitchen–dining–living loop; extra rooms extend that core.

```yaml
controls:
  room_layout:
    scene.room_count:
      randint: [3, 8]
    scene.room_arrangement:
      choices: [chain, branch]
    scene.extra_room_width:
      uniform: [3.5, 4.5]
    scene.extra_room_depth:
      uniform: [3.5, 4.5]
```

- **Room count:** 3–8 total, including the three core rooms.
- **Chain:** additional rooms extend east from the living room.
- **Branch:** an eastward spine with alternating north-facing leaf rooms.
- **Dimensions:** width/depth of all added rooms, each between 3 and 6 metres.
- **Furniture:** extensions use textured THOR armchairs, lamps, plants and
  occasional tables and rugs. Floors and walls inherit the existing living
  room materials and texture scale, keeping the extension consistent with the
  house. Leaf rooms group
  seating along the far wall; through rooms retain a central circulation zone.
  Meshes and collision shapes come from the local MolmoSpaces `train_18` asset
  catalog; its house layout is not imported. Set `CES_ROOM_FURNITURE_CATALOG`
  to override the catalog XML. Missing files produce a setup error.
  Default extensions are 5 × 5 m; smaller rooms substitute stools when larger
  seating does not fit. This remains a chain/branch extension, not a general
  whole-house floor-plan generator.
- **Doors:** each added connection has physical hinged leaves and handles.
  New doors start open by default. Door bindings discover all installed indices
  instead of assuming exactly three doors.

The scene builder creates real wall openings, frames, floors, wall segments,
and door lintels. It does not make closed walls or doors non-colliding. Existing
room names and task supports are preserved. Extra floor bodies use the naming
convention consumed by the existing navigation-map builder.

## Run

An example sampling file is provided:

```bash
# Preview five sampled configurations and their plans without running physics.
bash scripts/generate_sampled_composites.sh \
  --sampling-yaml configs/composite_sampling/examples/set_table_extra_rooms.yaml \
  --describe

# Generate those five independent examples.
bash scripts/generate_sampled_composites.sh \
  --sampling-yaml configs/composite_sampling/examples/set_table_extra_rooms.yaml
```

Combine `room_layout` with other groups in one YAML and adjust
`randomize_controls.min_active/max_active`. Counts remain attempts rather than
guaranteed successful episodes.

To close an extra door, use a fixed parameter such as
`scene.room_doors.room_door_3: closed`, or put it in the `door_route` group with
`choices: [open, closed]`. Keep `room_count` large enough for every specified
door; invalid indices are rejected. Door 3 connects living to extra_1; subsequent
door indices follow construction order. Selecting a core shortcut/detour sets
the first three door states together, preserving explicitly configured extra
door states. Extra closed doors require the planner to open them to enter those
rooms; the existing oracle recipes do not automatically add such calls.

## Metadata and skills

`room_graph.json` records room names, floor bodies, added-room centers and sizes,
connections, doorway centers, and furniture names. This is also included in the
final `task_manifest.json`. Door states remain in `room_doors.json` and the saved
scene configuration. All sampled parameters remain in `config.json`.

The existing household `navigate` interface accepts added room names such as
`extra_1`, using the shared checked navigation controller toward the room center.
Room doors retain ordinary `navigate`, `open_door`, and `close_door` bindings.
The two onboard cameras stay unchanged regardless of room count.

## Current scope and evidence

The five existing composite recipes still use their original kitchen and dining
supports. Added rooms change the environment and provide exploration destinations;
they do **not** yet relocate meal service, baskets, or appliances into those rooms.
That requires task-specific destination configuration and goal-check coverage.

Unit tests cover 3/4/8 rooms, both arrangements, door bindings beyond index 2,
invalid dimensions/counts, clear doorway centers, mixed sampling and shared room
navigation dispatch. Real-scene diagnostics compile the robot, generate the
existing collision occupancy map, and verify a shared free-space component
between living and the added rooms. Static map connectivity is not a physical
robot rollout or a guarantee that every randomized composite will succeed.

Diagnostic previews:

- [Four-room chain: top view](../runs/room_graph_preview_4_chain/top_view.png)
- [Six-room branch: top view](../runs/room_graph_preview_6_branch/top_view.png)
- [Six-room branch: 3D overview](../runs/room_graph_preview_6_branch/overview.png)

## Furnished extension preview

See `runs/furnished_extension_preview_4_chain` and
`runs/furnished_extension_preview_6_branch` for 3D/top views. Validation covers
model compilation and static navigation-map connectivity, not a composite
robot rollout. The original three-room core is preserved.
