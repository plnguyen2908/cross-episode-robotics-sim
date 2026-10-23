# Composite variation coverage demos

Run from the repository with the simulation environment activated:

```bash
bash scripts/generate_composite_variations.sh
```

This runs 35 independent episodes, seven for each of the five tasks, with two task workers and automatic GPU assignment. Each task runs its seven variations sequentially. With --workers 5, all five tasks start in parallel; two variations of the same task never run concurrently. Table setup and clearing use two people and clearing uses compact mugs. Every invocation creates a unique runs/composite_pilot_* folder. These are coverage demos, not an established train/test split or 35 guaranteed successes.

For the four cross-room tasks, the seven examples are baseline, alternate procedure, kitchen layout, furniture translation plus rotation, same-category object instances, closed-door shortcut, and open living-room detour. The last two share the same initial scene and goals but take different routes. Warm-two-pots remains kitchen-local and uses the existing seven axes: baseline, alternate procedure, layout, furniture translation, furniture rotation, object instances, and combined.

There are currently two task-specific procedures, not three independent procedure implementations. The alternate procedure covers execution-order variation; the route pair additionally changes the skill sequence for cross-room tasks. Relocating appliances or serving into a new destination room is not included in this preset.

Useful options (later arguments override script defaults):

```bash
# One task, or one person
bash scripts/generate_composite_variations.sh --task clear_and_store --people 1
# Change parallelism
bash scripts/generate_composite_variations.sh --workers 3
# Inspect all specifications without running simulation
bash scripts/generate_composite_variations.sh --write-configs /tmp/composite_variation_configs
# Validate dependencies without running episodes
bash scripts/generate_composite_variations.sh --preflight
```

The terminal shows overall tqdm progress and per-task stage changes. Each episode has config.json, run.log, skill_calls.json, and report.json; successful episodes include composite.mp4, head_camera.mp4, and action_recording/. Batch results.json records failures as failures and continues with the remaining episodes. Default timeout is 7200 seconds per episode; override with --timeout.

Generate trajectories with live atomic motion optimization and smoothing enabled, including legacy-config reruns. Optional further replay acceleration is a separate physical validation step; the source is preserved. New variations still require physical qualification. Reserve selected layouts, instances, and room settings for OOD evaluation when constructing a larger dataset.

## Pickup recovery

If a pick exhausts reachable grasps at its dock, it returns a failure without
hidden base movement. The composite oracle can explicitly call navigate with
another dock, then pick again, up to four recovery docks. Every attempt remains
in skill_calls.json; a recovered episode can contain failed attempts and still
pass its measured final goals. Top-down and qualified diagonal grasps are tried
at the current dock before repositioning. Nearby empty-hand recovery (within 1.5 m) reuses the door-redocking executor: collision-checked forward motion at at least 0.30 m/s and 0.8 rad/s turn speed, with settings restored afterward. Candidates matching the current pose are skipped; the tuck helper skips movement when already tucked. Longer moves retain normal navigation settings.
Planner-process errors (including Broken pipe) are not retried as grasp failures.
This change does not enable unqualified live timing experiments.

### Kitchen-layout failure regression

The layout-3 table run in composite_pilot_4idbb398 failed on bowl_01. Counter
allocation excluded any body with a joint, inadvertently ignoring hinged
appliance doors. Allocation now excludes free objects rather than articulated
fixture parts, and household source objects reserve 12 cm of vertical pickup
clearance. In composite_pilot_2r4vy25c the first bowl moved about 54 cm along the
counter and its pickup passed; the full episode was still running at this update.
Recovery/scene tests passed (72 tests in the recovery suite, 31 in the overlapping
scene suite). No full-episode success or acceleration validation is implied.

### Stationary grocery baskets and four-view videos

Grocery baskets are preplaced by datagen and stay at their initial locations.
Only the groceries are picked, carried, and placed. The two procedures sort
all drinks first or all snacks first. Configure locations with:

```json
"basket_supports": {
  "purple_basket": "kitchen_counter",
  "orange_basket": "dining_table"
}
```

Each support can be `kitchen_counter` or `dining_table`. The seven-demo suite
includes both in the kitchen, both in dining, and split-room placement; closed-door
route comparisons use dining baskets so the route choice remains meaningful.
The paired original/procedure examples retain identical scene setup. Furniture
and layout variants move the supports; datagen allocates supported, clear basket
footprints. Success requires correct color sorting and unchanged basket poses
(with the existing small physics-settling tolerance), not basket transport.
Baskets retain ordinary contact physics; no weld is added. Regenerate grocery
configs: old basket-carry procedures and goal contracts are intentionally obsolete.

All newly rendered composite videos use a synchronized 2×2 grid: kitchen,
dining, living, and the robot's current camera. The existing wall-corner room
cameras and robot camera are used, at normal simulated playback speed. Previously
rendered videos are not changed automatically.

## Continue an existing batch

After the previous generator has stopped, run:

```bash
bash scripts/generate_composite_variations.sh \
  --resume runs/composite_pilot_w5h4focu --workers 5
```

Successful examples with an existing, nonempty video are kept. Failed or unfinished
examples restart from scene setup using their saved configuration and the current
code. Previous attempts (including videos and logs) move to
`_previous_attempts/<episode_id>/<attempt>/`. Progress includes kept successes.
Tasks run in parallel; variations of the same task remain sequential.

Use `--task unpack_and_sort_groceries` to retry only that task. Saved people counts
take precedence over the shell wrapper's default. New batches save
`batch_configs.json` upfront so even examples never started can be resumed.
Older batches can resume only examples with an existing `config.json`.
Do not resume a folder while an older generator is still writing to it.
New generators lock the batch folder to prevent simultaneous resume processes.
This is example-level retry, not restoration halfway through a physical trajectory.


### Regenerate videos with the current motion policy

Resume keeps successful videos produced by older code. To regenerate all
variations using the current
[atomic timing defaults](atomic_duration_defaults.md), start a fresh batch:

```bash
bash scripts/generate_composite_variations.sh --workers 5
```

This creates a unique output folder. Existing results remain available.
