# Configurable history and current generation

The `cross_episode_sim.generation` API creates histories independently and reuses
successful histories when generating current examples. Both call the existing
physical workers and their atomic skill interfaces. This is the first
counting/imitation implementation of [the design](history_task_design.md).
Temporal cooking and preference tasks are not implemented; unsupported task names
raise an error before simulation.

## Python interface

```python
from cross_episode_sim.generation import (
    HistoryConfig, CurrentConfig, create_history, create_current,
)

coffee = create_history(
    HistoryConfig(task='coffee', seed=100, parameters={'cups': 1},
                  scene={'machine_room': 'kitchen'}),
    library='runs/history_library', gpu='0',
)
breakfast = create_history(
    HistoryConfig(task='breakfast', seed=101, parameters={'people': 3}),
    library='runs/history_library', gpu='1',
)
example = create_current(
    CurrentConfig(task='breakfast_with_coffee', seed=102, histories=(str(coffee), str(breakfast)),
                  scene={'machine_room': 'dining', 'extra_vessels': 3}),
    output='runs/composed_example', gpu='2',
)
```

Calls block until completion and return the actual output directory. Each
invocation creates a unique directory, including when the requested output
already exists. Save the returned history paths and pass them to any later
`create_current` call; history physics is not rerun. Each worker prints stage
progress and writes a log under `runs/`. Definitive physics failure stops the
worker, including its rendering subprocesses, rather than waiting for diagnostics.
Failure raises an exception and retains the attempt for inspection.

## JSON configuration and CLI

Ready-to-edit configs live in `examples/generation/`:

```bash
python -m cross_episode_sim.generation history \
  --config examples/generation/coffee.json --gpu 0
python -m cross_episode_sim.generation history \
  --config examples/generation/breakfast.json --gpu 1
```

Copy the printed history directories into `histories` in
`examples/generation/current.json`, then run:

```bash
python -m cross_episode_sim.generation current \
  --config examples/generation/current.json --gpu 2
```

History paths in JSON are relative to the config file, or absolute. Python API
paths are relative to the caller's working directory. Use `--library PATH` to
choose the history library and `--output PATH` to choose a current example's
output. `--dry-run` validates the configuration without loading histories or
running physics. The CLI includes a tqdm completion bar; worker logs show stage
progress during execution.

| Configuration | Meaning |
|---|---|
| `task` | History: `coffee` or `breakfast`; current: a registered task name (built-in: `breakfast_with_coffee`) |
| `seed` | Nonnegative integer controlling this episode's layout |
| `parameters` | Task-specific options: breakfast history `{"people": 1–4}`, coffee history `{"cups": 1}`; breakfast-with-coffee current `{}` derives its count from history |
| `histories` | Current only: validated history directories in presentation order; requirements depend on the task |
| `layout_profile` | `varied` by default; `baseline` uses the legacy baseline sampling profile |
| `scene` | Optional overrides listed below |

Scene options: `extra_furniture` (0–8), `floor_objects` (nonnegative),
`nav_speed` and `turn_speed` (positive), breakfast `table_objects`, coffee/current
`clutter` and `machine_room` (`kitchen`, `living`, `dining`), current
`extra_vessels` (2–6; default 3). Current uses the legacy fixed pool formula
`3 + extra_vessels` per vessel type: default six cups and six bowls, independent
of whether the required count is 1, 2, 3, or 4. The name is retained for worker
compatibility; it is not a count added to the number of people.

For `breakfast_with_coffee`, current serving count is derived from breakfast history and
cannot be overridden in its config. Other current tasks own their parameter rules. Unknown options are rejected. Legacy `baseline` does not mean all
scene variation is disabled: breakfast retains its original worker randomization.

## Records and compatibility

A history becomes reusable only after physical goal checks and readable video
endpoints pass. `history.json` stores its ID, configuration, category, compatibility
contract, validation evidence, and hashes of the report, task manifest, and video.
Loading a history checks those hashes and validation again. Keep library runs and
their scene assets available; this initial implementation does not make a portable
asset bundle or promise compatibility after simulator changes.

Current output retains `episode.json`, `run_manifest.json`, `prompt.txt`, the
numbered history videos, current video, and corresponding run directories. The
collector accepts one or more histories followed by current. History artifacts are hardlinked where possible, copied otherwise;
no history robot execution occurs. Treat published artifacts as immutable.
History IDs and source directories remain recorded for provenance. Use the normal
`collect_examples` pipeline to freeze evaluation initial scenes.

The built-in `breakfast_with_coffee` task accepts compatible one-cup coffee and
counting-breakfast histories. Custom current definitions choose their own history
requirements from the available history types.
The current contract uses the existing coffee-machine recipe and shared atomic
interface. It does not yet support multiple learned machine recipes, preference
composition, or temporal cooking. Generator metadata is not added to the planner
prompt. Avoid placing examples sharing histories in different dataset splits;
automatic split assignment is not part of this interface yet.

The existing `scripts.generate_examples` batch command remains unchanged and
still generates bundled episodes. This new API is an explicit separate path;
it does not silently change an existing batch or resume it with different
semantics. A failed current can be retried with a different seed and the same
history paths.

## Validation scope

Automated tests cover configuration errors, history integrity, failed-history
publication, reuse without rerunning histories, 1–4 count propagation, unique
outputs, and compatibility with the existing dataset collector. They use
synthetic worker outputs and do not establish robot success. Four-person scene
population and skill-plan support are added, but a successful full four-person
physical rollout is still required before treating that count as qualified.

## Adding differently named current tasks

`CurrentConfig.task` selects an executable task definition, while optional `name`
is a display label. `parameters` contains task-specific configuration. A task
owns its history requirements, scene options, prompt, execution, and goal checks;
there is no universal serving count or requirement for two histories.

Register a `CurrentTaskDefinition` with four callbacks:

| Callback | Responsibility |
|---|---|
| `validate_config(config)` | Reject unsupported parameters, scene options, and history counts before work starts |
| `compose(config, history_records)` | Check history compatibility and return `CurrentPlan(worker_task, settings, prompt, video)` |
| `execute(plan, output, gpu)` | Execute this task through atomic skills and write artifacts under `output/runs/current` |
| `verify(plan, run)` | Raise if task-specific measured goals are unmet |

The shared generator additionally requires a successful report, all reported
physical goal predicates true, and a complete readable video. A custom verifier
cannot bypass these checks. Use the standard run artifact format (`report.json`,
`task_manifest.json`, `adapter.json`, scene and grasp assets) for dataset collection.
Custom execution callbacks must also handle their subprocess cleanup and progress;
`run_episode` provides those for the existing breakfast/coffee workers. A new robot
task needs its own worker and evaluation environment adapter, not just a new name.

See the working plugin in [examples/current_tasks.py](../examples/current_tasks.py).
It registers `coffee_practice`, which takes **one coffee history**, accepts a
`cups` parameter, and brews cups back to their kitchen homes instead of serving
bowls and cups at the dining table:

```bash
python -m cross_episode_sim.generation current \
  --task-module examples.current_tasks \
  --config examples/generation/coffee_practice.json --gpu 0
```

Replace the example history path before executing. Add `--dry-run` to check the
config without simulation. For Python usage, import your registration module
before constructing `CurrentConfig(task='coffee_practice', ...)`. CLI
`--task-module` can be repeated. These are executable Python modules you supply.
Unknown task names and duplicate registrations raise errors.

The extension mechanism is tested using synthetic worker output, including a
single-history task with different behavior and a rejecting custom verifier.
The coffee-practice example uses the existing worker but has not received a new
physical rollout qualification as part of this interface change.

## Shared configuration shape

Both history and current configs use `task`, `seed`, `parameters`, and `scene`,
plus optional `name` and `layout_profile`. Current additionally supplies
`histories`. Task definitions are selected internally by `task`; callers do not
need to register the built-in breakfast-with-coffee task.

```python
coffee_config = HistoryConfig(
    task="coffee", seed=100, parameters={"cups": 1}, scene={})
breakfast_config = HistoryConfig(
    task="breakfast", seed=101, parameters={"people": 3}, scene={})
current_config = CurrentConfig(
    task="breakfast_with_coffee", seed=102,
    histories=[str(coffee_history), str(breakfast_history)],
    parameters={}, scene={})
```

The same fields are accepted in JSON. History parameters are checked by task;
unknown keys and conflicting legacy values fail before simulation. Newly written
history configs store counts inside `parameters`. Existing top-level history
`people` values and the old current task name `full_breakfast` remain readable.
The internal physical worker is still named `full_breakfast`; its behavior is
unchanged by this naming/configuration update.

## Video timing

New breakfast, coffee, and breakfast-with-coffee videos play at **1× simulation
time**, including stationary waits. They use the untrimmed accepted execution
trace, sampled at 25 frames per simulated second. Older saved controller settings
cannot restore the previous 3× speedup. Existing videos are unchanged.

The video shows the physical robot/object trajectory. Time spent computing
physics, planning, scene setup, and deferred rendering is wall-clock overhead,
not extra video duration. Rejected trials rolled back by atomic skills remain in
separate diagnostics, not the accepted demonstration. Failed-run diagnostic videos
may append an endpoint caption; successful demonstrations retain the recorded
simulation timeline (within frame sampling precision).

## Generate independent history batches

From the repository root, in your simulation environment:

```bash
COUNT=30 bash scripts/generate_history_library.sh
```

This starts two independent workers concurrently: **30 successful one-cup coffee
histories** and **30 successful breakfast histories total**, balanced across
1–3 people (10 successes each). Each worker runs its episodes sequentially and automatically leases an idle GPU
for each attempt, holding it until physics and rendering complete. Nothing is generated for current yet. This is
an actual simulation run and can take substantial time; four-person batches are disabled because initial kitchen placement repeatedly fails. Failed seeds do not count toward the target.

The shell shows two live tqdm bars on stderr, including success counts, active
seed, serving count, and attempt number. Detailed stage output goes to separate
stdout logs. The bars advance only when a history passes validation; a long
physical execution may stay at the same count. Watch individual stages with:

```bash
tail -f runs/history_batches_XXXXXX/coffee.log \
        runs/history_batches_XXXXXX/breakfast.log
```

Set `PYTHON_BIN=/path/to/venv/bin/python` if needed. Optional environment variables
`COFFEE_START_SEED` and `BREAKFAST_START_SEED` default to 100 and 1000. To resume,
pass the same directory (saved target and scene settings are retained):

```bash
bash scripts/generate_history_library.sh runs/history_batches_XXXXXX
```

Or launch either history type independently:

```bash
python -m scripts.generate_coffee_histories \
  --count 30 --start-seed 100 --gpu auto --output runs/coffee_histories
python -m scripts.generate_breakfast_histories \
  --count 30 --start-seed 1000 --people 1 2 3 --gpu auto \
  --output runs/breakfast_histories
```

Use `--resume DIRECTORY` in place of `--output` to continue. Omit `--output` to
create a unique batch folder. An existing explicit output is rejected to avoid
replacing work. `--scene-config FILE` accepts a JSON object of scene overrides.
`--max-attempts N` optionally caps attempts in that invocation; the default keeps
trying new seeds until the success target is reached. Ctrl-C preserves completed
histories and stops the active worker. Interrupted attempts advance to a new seed
on resume unless a validated history was already published and can be recovered.

Each task batch retains:

- `batch.json`: target, saved settings, next seed, and all attempts/errors.
- `histories.json`: successful history IDs, absolute paths, categories, task
  configs, sampled settings, compatibility contract, hashes and validation evidence.
- `attempt_XXXX_seedN/`: each attempt's original run and diagnostics, including its
  `history.json` if successful. Do not delete the library assets after indexing.

These indices are private generation metadata for later mix-and-match. For example:

```python
import json
from pathlib import Path
from cross_episode_sim.generation import CurrentConfig, create_current

batch = Path('runs/history_batches_XXXXXX')
coffee = json.loads((batch / 'coffee/histories.json').read_text())['histories'][0]
breakfast = json.loads((batch / 'breakfast/histories.json').read_text())['histories'][0]
example = create_current(CurrentConfig(
    task='breakfast_with_coffee', seed=2000,
    histories=[coffee['path'], breakfast['path']], parameters={}, scene={}),
    output='runs/composed_example', gpu='2')
```

For breakfast-with-coffee, final output keeps the existing datagen layout:
`episode.json`, `run_manifest.json`, `prompt.txt`, `history_1.mp4`, `history_2.mp4`,
`current.mp4`, and `runs/{history_1,history_2,current}/`, plus configuration and
validation metadata. Existing collection/evaluation loading continues to work.
The histories are reused; only current physics is executed for each combination.

### Automatic GPU assignment

The batch scripts default to `--gpu auto`; the shell launcher needs no per-task
GPU variables. Before each attempt they query `nvidia-smi`, restrict candidates
to `CUDA_VISIBLE_DEVICES` when set, and choose an idle GPU with at least 8 GiB
free memory, at most 10% utilization, and no listed compute process. Cooperative
file locks keep these history workers from choosing the same GPU. The selected
UUID is saved in `batch.json`. A waiting worker displays `waiting for GPU`.

With only one eligible GPU, workers take turns. Use `CUDA_VISIBLE_DEVICES=0,1,2`
to restrict the automatic pool, or the Python scripts' `--gpu INDEX` for an
explicit assignment. External applications do not participate in these locks,
so availability checks cannot reserve a device against unrelated jobs.

Breakfast batch generation is currently limited to **1–3 people**. Resuming an
older 1–4 batch automatically removes four from future selections, retaining all
completed histories and failed-attempt diagnostics. Existing low-level four-person
scene/config support remains experimental; it is not used by batch generation.

### Live stage progress

Each batch bar now includes the active seed, number of people, atomic step count,
status, and latest detailed stage, refreshed about every 10 seconds. For example:

```text
coffee successful histories: 3/30 ... seed=108 people=1 [3/16] running | pour grounds
breakfast successful histories: 4/30 ... seed=1010 people=2 [0/?] preparing | scene setup
```

Statuses include waiting for GPU, preparing, running/retrying, rendering,
validating, completed, and failed. A `?` total means the scene's skill plan is not
available yet. Rendering is shown as a phase, not a measured frame percentage.
Each task's `progress.json` records the latest status and full stage text; its
stdout log retains the stage history. Existing running Python launchers need to
be stopped and resumed to pick up this display change.

## Sample current tasks from history batches

```bash
python -m scripts.sample_current_tasks \
  --config examples/generation/sample_breakfast_with_coffee.json \
  --history-batches runs/history_batches_BbuBNu \
  --count 1 --sampling-seed 7 --output runs/current_selections
```

This command only selects histories and writes configs. Add `--execute` to run
current physics on an automatically assigned GPU and assemble the complete
existing datagen example. Without it, no robot execution is started. Supply
multiple batch directories (or individual `histories.json` indexes) to combine
libraries. Selection reads each index once per requested example; incomplete
attempts absent from successful-history indexes are not candidates.

The config's `task` selects the registered task's dependency rules. Built-in
`breakfast_with_coffee` requires a one-cup coffee demonstration and a breakfast
history matching `parameters.people` (1–3). Omit `people` to sample any supported
count and derive current's count from the selected history. Coffee compatibility
currently uses the existing `breakfast-coffee-v1` contract; no separate fictional
procedure ID is inferred for old histories.

Optional `history_requirements` adds filters per dependency slot. For example:

```json
{
  "task": "breakfast_with_coffee",
  "seed": 2000,
  "parameters": {"people": 2},
  "scene": {},
  "history_requirements": {
    "coffee": {"settings": {"machine_room": ["kitchen", "living"]}},
    "breakfast": {"parameters": {"people": 2}}
  }
}
```

Nested objects match the listed fields; lists permit any listed value. Filters
may narrow the task's dependencies but cannot replace them with conflicting
requirements. `scene` controls the new current layout independently; filtering
coffee history to the kitchen does not force current's machine into the kitchen.
No matching history produces an explicit error rather than an incompatible pair.
Selected histories undergo the normal video, goal, and integrity validation;
the task's composition check also runs before publishing a current config.

Each `selection_XXXX/` contains `current_config.json` and private `selection.json`
with the sampling seed, source indexes, slot-to-history IDs/paths, constraints,
eligible counts, and hashes. With `--execute`, it additionally contains an
`example/` in the normal datagen layout and `result.json`; the example also stores
`history_selection.json`. Selection metadata is not injected into planner prompts.

`--count` counts selections, not successful physical runs. A failed execution
stops this script and retains diagnostics. Histories are sampled with replacement
across selections; reuse is intentional, and exact pair repeats are possible.
Seeds are incremented for current layouts and selection RNG separately. Repeating
with the same inputs and unchanged library reproduces the selected histories.

New current definitions can supply a `history_requirements(request)` callback in
`CurrentTaskDefinition`; use `--task-module` to load their registration. The
`coffee_practice` plugin already supports this and requires only a coffee history.
History generation currently supplies coffee and breakfast records; adding a new
current name does not create unsupported history types automatically.

Shell wrapper (samples histories AND executes current by default; 30 attempts with 1–3 people inferred from the sampled breakfast histories):

```bash
bash scripts/sample_current_tasks.sh \
  --history-batches runs/history_batches_BbuBNu \
  --output runs/current_selections
```

The shell wrapper always enables execution and writes complete datagen examples
under `selection_XXXX/example/`, including both history videos and current video.
The Python CLI remains available for selection-only use. All Python CLI options are forwarded;
`--config FILE`, `--count N`, and `--sampling-seed N` override defaults. Paths are
resolved from your working directory. Set `PYTHON_BIN` to select your environment.

### Faster navigation trial

`sample_breakfast_with_coffee_fast.json` requests 0.36 m/s mean base travel and
0.8 rad/s mean turning, versus the default 0.24 m/s and 0.6 rad/s. Use it explicitly:

```bash
bash scripts/sample_current_tasks.sh \
  --config examples/generation/sample_breakfast_with_coffee_fast.json \
  --history-batches runs/history_batches_BbuBNu --count 1
```

This profile has configuration validation only; loaded-cup/bowl physical
qualification and measured speedup are pending. It does not speed up arm contact
motion or change grip, collision, and endpoint checks. Segment minimum durations
and slow reverse undocking remain in force. Existing workers keep their settings.

### Assemble a small test set from existing successful currents

```bash
bash scripts/mix_match_test_set.sh
# Override inputs or size:
python -m scripts.mix_match_test_set \
  --currents runs/current_examples_sb9eo4qc \
  --history-batches runs/history_batches_BbuBNu \
  --count 5 --sampling-seed 0 --output datasets/test_set_v1
```

The shell defaults select the two completed currents in `current_examples_sb9eo4qc`
and histories from `history_batches_BbuBNu`. The script validates existing currents,
round-robins them, and samples distinct compatible combinations using task dependency
contracts and the recorded serving count. Coffee supplies a one-cup machine-use
demonstration; breakfast must match the current's number of people. Current settings
must exactly match the recomposed configuration. Physics and rendering are not rerun.

The output has `dataset.json` for `scripts.evaluate_dataset`, plus complete example
folders containing history/current videos, prompts, runs, metadata, oracle reports,
and frozen initial scenes. `history_selection.json` and `provenance.json` record
source choices. Files are hardlinked when possible; treat source and assembled
artifacts as immutable. Existing output directories receive a unique suffix.
Progress bars cover validation, matching, and assembly.

Five combinations from two currents are **two independent current scenes**, not five.
The index records `unique_currents` and `current_group` so downstream splits and
analysis can group repeats. `layout_fingerprint` includes histories as in the
existing dataset collector; use `current_group` to identify reused current scenes.
The current implementation supports the breakfast-with-coffee serving-count contract.

### History EQA alongside action tests

Mix-matched examples now include `eqa_questions.json` and `eqa_answers.json`.
Each example has three single-answer multiple-choice questions: required breakfast
bowls, required coffee cups (one per person), and the coffee machine's room in the
**coffee history**, not its current room. Counts come from the matched breakfast
history and must agree with the current serving count. Room answers are checked
against the coffee history's scene manifest. Choices are deterministically shuffled.

Only `eqa_questions.json` and its referenced history videos are model inputs;
`eqa_answers.json` contains private scoring labels and evidence. The two files live
in the same example directory as the action task. To add/update them for an existing
assembled dataset without rerunning simulation:

```bash
python -m scripts.create_eqa --dataset datasets/test_set_v1
```

The first version has five examples / fifteen questions. Since both source currents
serve one person, every count answer in this small dataset is one; broader count
coverage requires currents with two and three people. This adds QA data only;
scoring model QA responses is separate from action-task evaluation.

Evaluate EQA with the running Qwen native-video server (no simulator needed):

```bash
bash scripts/evaluate_eqa.sh --endpoint http://127.0.0.1:8020/v1
# Or: python -m scripts.evaluate_eqa --dataset datasets/test_set_v1
```

This submits all three questions together with the referenced complete history videos
for each example. The model processor owns video sampling; `--video-fps` is an optional
explicit override. Requests default to a 1,800-second timeout (`--request-timeout`).
Outputs go to a fresh `runs/qwen_eqa*` directory with raw responses, per-question
predictions and `summary.json` containing accuracy overall and by question/category.
Missing or invalid choices count as incorrect. Request failures are reported separately
and stop the run, marking the summary incomplete rather than presenting a partial
score as full-set accuracy. Private answer keys and current rollout videos are never
included in requests. No EQA inference is launched by dataset generation itself.
