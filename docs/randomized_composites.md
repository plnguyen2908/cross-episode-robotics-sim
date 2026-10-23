# Randomized composite generation

The default `scripts/generate_composite_variations.sh` now selects `--suite randomized`:
seven independently sampled configurations for each of five tasks. This is no
longer the fixed seven-case coverage suite. Each task has its own editable YAML in
`configs/composite_sampling/randomized/`.

```bash
bash scripts/generate_composite_variations.sh --workers 8
# A different reproducible batch, 100 attempts per task:
bash scripts/generate_composite_variations.sh --workers 8 --seed 123 --count-per-task 100
# Inspect generated configs without executing:
python -m cross_episode_sim.generation.composite_pilot --suite randomized --write-configs /tmp/my_configs
# Customize one or more per-task YAML files:
bash scripts/generate_composite_pilot.sh --sampling-yaml configs/composite_sampling/randomized/set_table.yaml --workers 1
```

Scheduling uses a shared **episode queue** with eight workers by default.
Variants of the same task can run concurrently. Each episode exclusively leases
one idle visible GPU through execution and rendering; the next queued episode
uses whichever GPU becomes available. Assignment is printed and saved in
`gpu_assignment.json`. Fewer pending episodes or idle GPUs means fewer active
workers. `CUDA_VISIBLE_DEVICES` can restrict the pool, and `--workers` overrides
the concurrency limit. Resume keeps completed examples.

Use a NEW batch to get randomized houses. `--resume runs/EXISTING_BATCH` preserves
its saved configs—including old fixed cases—and skips successful examples. It
never silently changes the distribution of an existing dataset. The old coverage
suite remains explicitly accessible with `--suite variations`.

## Configuration and execution

1. Sample an independent house seed and room count **before** drawing the control
   subset. Default: 5–7 rooms, exactly one kitchen/dining and at least one bedroom.
2. Draw any subset of available controls (`min_active` through `max_active`,
   including zero or one). Draw each selected parameter from its distribution.
3. Build the selected native kitchen; fit a constrained-growth house around it.
   This geometry depends on the selected kitchen bounds; the house seed is
   independent of the variation selection. Validate base furnishings and doors.
4. Apply furniture translations/rotations and validate again. Invalid candidates
   are rejected, with seed and reason saved. Search up to `base_layout.max_attempts`;
   after 25 and 50 rejections, the room-area target grows by 20% and 40%
   respectively, without changing room count or weakening clearance checks.
   Layout sampling is conditional on the selected
   task controls being feasible. It does not guarantee physical task success.
5. Resolve graph routes and initial door states. Shortcut closes an edge only if
   an alternate kitchen-to-dining path exists. Shortcut policy opens the necessary
   leaves using atomic calls; detour policy leaves the edge closed. A house with
   no such cycle is rejected for these controls. Door indices are not hard-coded.
6. Add task objects/appliances, apply appearance, and validate final furnished
   circulation, open-door connectivity and initial robot contact. Exact carried-arm
   clearance/contact feasibility remains checked during atomic execution.
7. Execute the task through the atomic interface with the existing mandatory
   motion optimization, smoothing and retiming. Check the task goal and save
   actions, actual skill order, metadata, and 2×2 navigation/wrist RGB-D video.

Scene authoring retries up to `scene.base_layout.setup_attempts` (default 3,
maximum 10) for exhausted house sampling or unsupported object-placement sites.
It preserves task controls, kitchen seed, and selected asset instances; only
the base-house seed changes deterministically. Each rejection is recorded in
`scene_setup_attempts.json` and its scene/layout evidence in `rejected_setups/`.
Unknown configuration errors fail immediately. Exhausting all retries still
fails explicitly; retries never count as successful rollouts. This retry loop
covers scene authoring, not controller initialization or failed robot actions.

The stove task allocates counter source positions and uses burner destinations,
without reserving unused dining-table slots. Rotated-table allocation searches inward from the
world bounding-box edge and retains full footprint support-ray checks.

Setup regression checks (October 21): both failed pot configurations from
`composite_pilot_k8yxo0si` now complete real scene authoring and object allocation.
Example 1 accepts a seven-room house on candidate 30 at the expanded area target
(`runs/pot_setup_fix_1_dpc0tsda`); example 2 retains its original accepted house
and allocates both pots on the kitchen counter (`runs/pot_setup_fix_2_z_n2gr5_`).
Twelve focused tests pass, including rotated support allocation, deterministic
bounded retries, and room geometry. These checks do not constitute full robot
rollouts or certify controller initialization.

## Controls

- Execution order: task-specific supported procedures.
- Object instances: qualified same-category pools; scene seed varies per example.
- Native kitchen: configurable IDs (defaults 1 and 3; builder currently supports 1–10).
- Furniture translation/rotation: continuous ranges. Dining furniture transforms
  as a group. Background controls select matching semantic furniture in sampled rooms.
- Furniture instances: seeded changes of living-room furnishing profiles (sofa
  lounge, reading, conversation, media) and choice among matching catalog assets.
  The available diversity depends on that catalog; an asset type with one candidate
  cannot become a different instance.
- Door/route choice: default, open shortcut, or open-path detour. Kitchen-local
  warm-two-pots omits this redundant control.
- Appearance: coordinated themes for wall/floor/furniture/light plus independent
  RGB controls. `rgb_uniform: [[r_min,g_min,b_min],[r_max,g_max,b_max]]` supports
  continuous colors. Task-defining basket/sorting colors remain protected.

`count` means attempted examples, not guaranteed successes. Change `seed` for a
new reproducible batch. `room_area` (default 36 square metres) biases grown-room
sizes so furnished door approaches can fit; exact room areas differ. `rooms`,
`loop_probability`, and `max_attempts` tune architectural sampling. Kitchen
location remains anchored; surrounding polygons, roles, connections and furniture
are sampled. Imported houses are not used.

## Evidence saved per episode

- `config.json`: sampling specification, selected controls and parameter values.
- `base_layout.json`, `room_graph.json`: accepted polygons, connections, furnishings.
- `base_layout_validation.json`: seeds, rejection reasons, before/after circulation.
- `dining_transforms.json`: actual moved furniture bodies and transforms.
- `resolved_route.json`, `room_doors.json`: actual route and initial door states.
- `final_layout_validation.json`: final static checks (not a rollout certificate).
- `oracle_plan.json`, `skill_calls.json`, `report.json`: planned order, executed calls,
  and measured outcome. Successful static previews alone are never marked task successes.

This change does not revalidate all 35 combinations. Physical results are reported
separately from unit tests and scene checks.

## Integration validation, October 21

- 32 focused sampling, route, geometry and camera tests pass.
- Mixed geometry/appearance compilation check passed: actual dining transforms and
  colors changed, and circulation remained valid (`runs/randomized_mixed_geometry_check`).
- Full set-table execution passed in a newly generated six-room house, including
  kitchen-to-dining carries and the task evaluator. The four-panel video is saved:
  [report](../runs/randomized_pipeline_retry_2pwy_xdr/report.json),
  [video](../runs/randomized_pipeline_retry_2pwy_xdr/composite.mp4).
- A kitchen-edge shortcut diagnostic opened one leaf but failed local repositioning
  on the other. Route selection now prefers an eligible inter-room edge over an
  appliance-lined kitchen edge when both have an open alternative path. Both
  leaves on that sampled interior edge passed their atomic open postconditions.
  This preference is not a guarantee that every sampled door interaction succeeds.
- Final room navigation targets are chosen in checked free space instead of blindly
  using room centers (which can coincide with the dining table).

The smoke test uses a one-person task on a sampled house; it does not certify all
five tasks, all seeds, or the entire 35-example batch. Opening the two tested leaves
required about 25–26 simulated seconds each, exceeding the usual motion target.

The door-prefix diagnostic is saved at
`runs/randomized_graph_route_qjl4a3_i/results.json` (first four calls passed).
Its subsequent room navigation still used the old occupied center target and
failed. Repeating that phase from its saved failure state with the new free-space
room target passed in 13.68 simulated seconds:
[room navigation result](../runs/sampled_room_nav_ftw5fzc3/result.json).
This is a phase regression check, not a claim that the entire closed-door
composite was rerun end-to-end after that fix.

## Saved 3D house overview

New episodes save `house_overview.png` (1600×1000, elevated oblique view) and
`house_overview.json` immediately after saving the initial state. This is a
human diagnostic image, never an additional planner/VLA camera. It includes
actual initial furniture, objects, and door poses.

Backfill missing images for successful examples without re-executing physics:

```bash
python scripts/render_house_overviews.py runs/YOUR_BATCH
```

The command also accepts individual episodes or multiple batches. It skips
existing images and unsuccessful episodes, loads `model.mjb` and
`initial_state.npz`, and does not modify videos or success reports.
