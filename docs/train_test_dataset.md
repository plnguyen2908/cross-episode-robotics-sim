# Duration-budgeted training and testing pilot

This dataset uses a shared training corpus, not paired history/current packages.
It retains the five composite tasks and fixed atomic API. The targets are **60
minutes training + 15 minutes two-room testing + 15 minutes multi-room testing
per task**
of successful, normal-speed video. They are not generation wall-time limits.
Each task has an independent fine-tuning/test dataset and duration quota. Across
five tasks this is at least **5 hours training + 2.5 hours testing**. Whole episodes
are retained;
coverage requirements and the final episode can exceed the requested minutes.

## Split controls

| Control | Training | Two-room test | Multi-room test |
| --- | --- | --- | --- |
| Rooms | Kitchen + dining | Kitchen + dining | 5–7, including kitchen/dining/bedroom |
| Kitchen | Layout 1 | Layout 3 | Layout 3 |
| Asset selection | Baseline qualified pool | Alternative qualified pool | Alternative qualified pool |
| Dining translation | ±10 cm | ±30 cm | ±30 cm |
| Dining rotation | ±5 degrees | ±20 degrees | ±20 degrees |
| Appearance | Warm home / cool daylight | Sage neutral | Sage neutral |
| Procedure | Both supported procedures per task | Both | Both |
| Room doors | Closed direct door and open direct door | Closed/open direct door | Shortcut opening, detour, default |

The two-room profile is an explicit exception to the normal bedroom requirement.
Its kitchen connects directly to dining. A closed direct connection is opened
through ordinary `navigate`/`open_door` calls; it does not require an impossible
alternate route. Existing growth-house shortcut/detour semantics remain unchanged.
Kitchen-local `warm_two_pots` does not gain artificial room-door operations.
The other four training tasks must each contain a successful room-door-opening
example and an open-route example. Procedure and door schedules are crossed.
For grocery sorting, the current baseline baskets remain on kitchen supports;
its route-opening prefix is recorded but is not evidence that sorting itself
requires cross-room transport.

Larger test ranges overlap the training ranges: this is a changed distribution,
not an assertion that every coordinate is outside training support. Kitchen and
appearance choices are held out. This initial recipe combines shifts rather
than providing a complete set of single-factor ablations.

**Asset leakage audit:** alternative pools can share assets, and authored baskets
or boxes are intentionally reused. `asset_overlap.json` reports actual asset IDs
shared with training. Do not claim disjoint-object OOD from the pool name. Review
this audit and define stricter asset-key partitions before reporting that metric.
The sampled seed domains are separate; this alone is not proof of every form of
semantic independence. These are known-task/new-setting splits, not held-out tasks.

## Commands

Run in the simulation environment, or set `PYTHON=/path/to/sim/bin/python`.

```bash
# Write the first five configs per split and empty manifests; no simulation.
bash scripts/generate_train_test.sh --output runs/train_test_v1 --prepare-only

# Generate. Repeating this command resumes the same dataset.
bash scripts/generate_train_test.sh --output runs/train_test_v1 --workers 8

# Generate just training, or refresh inventory from already completed examples.
bash scripts/generate_train_test.sh --output runs/train_test_v1 --split train --workers 8
bash scripts/generate_train_test.sh --output runs/train_test_v1 --collect-only
```

The driver uses a **continuous shared episode queue**, with no task lanes or
round barriers. `--workers 8` allows eight concurrent examples, including eight
of the same task when needed. Each finished episode frees a slot immediately;
the coordinator records its result and dispatches another pending/new example
without waiting for the others. GPU leases use whichever visible GPU is available.
Task labels only balance successful duration and procedure/door coverage.

Failed payload folders are removed after worker completion; compact failure
records and original configurations remain. A fresh seed replaces a failed
example. Saved unfinished examples enter the same shared queue on resume.
Generation has no maximum attempt count and continues until duration/coverage
is met. Already-running examples finish and are retained, so duration can overshoot
by multiple episodes. Per-episode `--timeout` remains to bound stuck workers.
New attempts are stored under `batches/round_continuous/` (the name preserves the
existing collector format; it is not an execution round). Old numbered batch
folders remain readable and are not regenerated.

Resume verifies the saved seed, duration budgets and YAML contents. To change
distributions, create a new output directory. Configured durations are successful
1× video length, including legitimate waits, with no padding or video speedup.
A missing layout preview/object manifest prevents an episode from counting toward
the dataset quota, even if its physical report says success.

## Saved dataset structure

```text
train_test_v1/
  dataset_config.json            seed, budgets, exact source YAML contents
  asset_overlap.json             actual train/test shared asset IDs
  train/                         also test_two_rooms/ and test_multi_room/
    manifest.json                successful episodes, durations, coverage, paths
    status.json                  quota/coverage completion
    objects.json                 asset IDs, categories, episode membership
    objects.md                   readable object inventory
    layouts/                     copied 3D house overview per accepted episode
    batches/round_0000/
      batch_configs.json         exact replayable episode specifications
      results.json               batch success/failure outcomes
      round_finished.json        completed-attempt marker
      TASK_000000/
        config.json              sampled parameters and split provenance
        instruction.txt          task prompt
        oracle_plan.json         planned sequence
        skill_calls.json         actual sequence, outcomes and timings
        asset_selection.json     selected physical assets, including background
        initial_snapshot.json    object roles, categories, poses and supports
        base_layout.json         accepted room geometry
        resolved_route.json      initial door states and selected route
        house_overview.png       diagnostic 3D image
        model.mjb / initial_state.npz / scene.xml
        report.json / run.log / trace.json / action_recording/
        composite.mp4            nav RGB/depth above wrist RGB/depth at 1×
```

The episode index includes each object's actual ID, category, initial pose/support
where present in the snapshot, and relative links to its config, prompt, video and
executed sequence. The entire root can be moved without breaking index paths.
Absolute asset references inside simulator XML still require the installed asset
bundle; the index is portable, not a self-contained asset redistribution.

## Validation and limits

Focused tests cover deterministic split configurations, two-room connectivity,
closed-door atomic prefixes, coverage gates, and exclusion of failed or
metadata-incomplete episodes. Static scene authoring on an existing native kitchen
passed furnished circulation before/after transforms on the first candidate:
`runs/two_room_static_q36bu0sy` (two rooms, accepted seed 1345723984).

This is **not full physical qualification of the new two-room profile**. No full
training/test collection was launched during implementation. Robot motion,
fixture interactions and the final goal still need to pass for each generated
example. Existing 35/35 multi-room evidence does not certify these new settings.
Mandatory motion and camera-collision requirements remain enabled. Training VLA
export still requires checking action convention and observation synchronization.


### Launcher correction (October 23)

The initial train/test launcher omitted `ROBOCASA_DIR`, unlike the composite
launcher. The inspected run had 809 recorded failures with the same missing-root
error, before robot execution. Both shell and Python entrypoints now initialize
the asset root, and runtime prerequisites are checked before replacement sampling.
The old one-example-per-task rounds also capped concurrency at five; new rounds
fill the requested worker slots. Seven focused dataset tests pass. The faulty
active loop was interrupted; its failure logs and saved configurations remain.


### Failed episode cleanup

Dataset runs delete failed episode payload folders after the parent records worker
completion. Resume also cleans old failures and skips finished rounds without
printing a zero-minute message for each one. Compact `failures/EPISODE.json`
records retain the config, outcome and log tail; batch configs preserve attempt
IDs and seed provenance. Active workers, successful episodes and physical successes
with rendering errors are preserved. This policy applies to train/test generation;
it does not delete failed folders from unrelated composite batches.


### Continuous scheduling correction

The initial driver used one task example per round, then fixed-size rounds; both
were incompatible with the requested continuously filled GPU pool. They have
been replaced by `continuous_queue.run_continuous_queue`. Regression tests verify
that eight same-task examples run together and that a fast worker's slot refills
before a slower peer finishes. This scheduling change preserves saved configs and
successful episodes and does not change atomic motion policies.


## Per-task fine-tuning datasets

Use `tasks/TASK/train/manifest.json`, `tasks/TASK/test_two_rooms/manifest.json`
and `tasks/TASK/test_multi_room/manifest.json` as the task-specific dataset indices.
Each contains only that task's successful episodes, its own quota, coverage and
relative paths to configs, prompts, actual calls, layout images and videos.
Adjacent `objects.json` lists only assets used by that task and split. Physical
recordings stay in shared split storage to avoid copying large action/video files.

The shared GPU queue can mix tasks; this does not mix their training/evaluation
indices or duration accounting. One task's extra minutes cannot satisfy another's
quota. Progress shows both aggregate and per-task minutes.

Older saved settings are upgraded on resume from shared to per-task duration
budgets, with the previous config archived in
`dataset_config_before_per_task_budget.json`. Existing episodes/configurations are
kept and count toward their own task's larger quota. No scene distribution changes.
A running Python driver must be restarted to load the new accounting.
