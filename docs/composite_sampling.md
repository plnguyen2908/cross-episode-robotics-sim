# Configurable composite variations

Each task has a YAML in `configs/composite_sampling/`. Controls compose; they
are not mutually exclusive seven-episode categories. All outputs use the normal
composite generator, atomic skills, physical success checks, motion defaults,
normal-speed navigation/wrist rendering, GPU leasing, and resume behavior.

```bash
# Inspect sampled prompts and atomic plans without running physics:
bash scripts/generate_sampled_composites.sh \
  --sampling-yaml configs/composite_sampling/set_table.yaml --describe

# Run one task:
bash scripts/generate_sampled_composites.sh \
  --sampling-yaml configs/composite_sampling/set_table.yaml

# Run all five tasks concurrently (variants within a task remain sequential):
bash scripts/generate_sampled_composites.sh \
  --sampling-yaml configs/composite_sampling/*.yaml --workers 5

# Resume the printed OUTPUT directory; sampled configurations are preserved:
bash scripts/generate_sampled_composites.sh --resume runs/YOUR_BATCH --workers 5
```

Use the simulation environment's Python, or set `PYTHON=/path/to/python`.
To export configs without physics, use `--write-configs /path/to/new_configs`.

## YAML controls

- `task`: registered composite name.
- `seed`: sampler seed, reproducing the complete set of sampled configurations.
- `count`: number of **attempts**, not guaranteed successes.
- `randomize_controls: {min_active: 0, max_active: 6}` selects a random subset
  of the six groups per episode. Zero means the fixed baseline; one is a
  single-control variation; six selects all groups. `count` is always the total.
- Groups: `execution_order`, `object_instances`, `kitchen_layout`,
  `furniture_translation`, `furniture_rotation`, `door_route`. Each contains
  its dotted parameter paths. Translation groups dx/dy across furniture; rotation
  groups yaw values. The stove task has five applicable groups (no door route).
- Active means selected for sampling; a sampled value can equal its baseline.
  Choose alternative-only values if a selected group must produce a difference.
- `fixed`: literal values applied to the baseline.
- `controls`: dotted config paths with literal values or distributions.

```yaml
task: set_table
seed: 42
count: 30
randomize_controls:
  min_active: 0
  max_active: 2
fixed:
  scene.seed: 7
  scene.people: 2
controls:
  execution_order:
    procedure:
      choices: [by_place_setting, by_object_category]
  door_route:
    room_route:
      choices: [open_shortcut, living_room_detour]
```

`randint` endpoints are inclusive; `uniform` samples real values. Explicit
`choices: [1, 3]` restricts layouts; `choices: all_layouts` discovers positive
layout IDs from the installed RoboCasa scene registry rather than assuming
all integers up to a maximum are supported. Registry membership does not
mean a layout has been physically qualified for this task.

Additional fixed parameters: `scene.kitchen_style`, `scene.seed`,
`scene.asset_instance_set` (`baseline` or `same_category_alternatives`),
`scene.minimum_heating_seconds`, and each basket's support
(`scene.basket_supports.purple_basket` / `orange_basket`). Furniture names:
`dining_set`, `sofa`, `coffee_table`, `armchair_left`, `armchair_right`; each
accepts `dx`, `dy`, and `yaw_deg`. Unknown control paths are rejected.

The selected route jointly determines door states: `default` retains the
baseline open doors; the two route alternatives start with the direct door
closed and both living-room connections open. Stove tasks reject these route
alternatives because they stay in the kitchen. People controls update both
paired object counts and the prompt. Task-specific preflight runs before GPU
allocation. The saved config includes the YAML specification, sampled values,
active controls, scene hash, and generated prompt. The normal episode folder
also records the actual skill calls, reports, and videos.

## Scope and validation

Procedure sampling chooses the existing task-specific oracle procedures; it
does not yet synthesize arbitrary valid permutations. Object sampling chooses
the existing instance sets rather than arbitrary assets. The supplied files
start with layouts 1 and 3, but arbitrary mixed combinations have not been
physically qualified. Scene setup and execution can reject infeasible samples;
failed attempts remain recorded for inspection/resume. This interface does not
yet replenish failures until a successful-example quota is reached, guarantee
balanced coverage, or establish train/test splits. No long simulation batch is
started automatically when editing these YAML files.

## Onboard observations and extended controls

New renders and evaluation defaults use navigation + wrist cameras, not fixed
room views. Optional `furniture_instances`, `wall_color`, and `object_color`
groups extend the original six; see [onboard observations](onboard_observations.md)
for YAML examples, physical validation limits, and room-layout extensions.

Optional `room_layout` sampling is documented in [room-layout variations](room_layout_variations.md).
