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
    HistoryConfig(task='breakfast', seed=101, parameters={'people': 4}),
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
    task="breakfast", seed=101, parameters={"people": 4}, scene={})
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
