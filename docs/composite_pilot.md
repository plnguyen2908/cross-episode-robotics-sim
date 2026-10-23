# Independent composite pilot

The five composite policies and their seven variants execute through one persistent
physical environment. Datagen constructs the scene; policies call only the fixed
atomic interface. **Physical qualification of the full pilot is still in progress.**
Preflight checks installed prerequisites, not task success. Do not count generated
specifications or symbolic tests as physical demonstrations.

## Ownership

| Component | Responsibility |
| --- | --- |
| Datagen | Select objects/layouts/furniture, construct the scene once, initialize state, bind skills, save setup metadata |
| Environment | Maintain one continuous simulator state and apply ordinary world processes |
| Shared atomic skills | Act on the existing scene using the fixed interface; never create/reset a training scene |
| Composite policy | Produce a prompt and ordered atomic calls; never access simulator state or evaluate its own success |
| Evaluator | Independently measure the task goal and private qualifying durations |

[HouseholdCompositeTask](../cross_episode_sim/tasks/composites/base.py) shares policy
initialization and dispatch. Each subclass implements **only** `make_prompt()` and
`make_plan()`:

| Task | Policy class | Alternative procedures |
| --- | --- | --- |
| Set table | [SetTable](../cross_episode_sim/tasks/composites/set_table.py) | Complete each setting / bowls then cups |
| Clear and store | [ClearAndStore](../cross_episode_sim/tasks/composites/clear_and_store.py) | Finish one storage unit / open both, store, close both |
| Prepare and serve meal | [PrepareAndServeMeal](../cross_episode_sim/tasks/composites/prepare_meal.py) | Retrieve drink/snack during heating / before heating |
| Sort groceries | [UnpackAndSortGroceries](../cross_episode_sim/tasks/composites/sort_groceries.py) | Fill baskets then move / move baskets then fill |
| Warm two pots | [WarmTwoPots](../cross_episode_sim/tasks/composites/warm_pots.py) | Start first burner before placing second pot / place both then start |

The base receives a `FixedSkillInterface`, not an environment. Execution reports
whether calls completed; it cannot declare the composite goal achieved. An oracle
recipe and a VLM planner submit the same action dictionaries to the same interface.

Scene specifications live in [composite_scenes.py](../cross_episode_sim/generation/composite_scenes.py).
Physical scene construction lives in [household_scene.py](../cross_episode_sim/generation/household_scene.py)
and [household_native_scene.py](../cross_episode_sim/generation/household_native_scene.py).
Goal predicates live in [composite_goals.py](../cross_episode_sim/evaluation/composite_goals.py).
The persistent simulator adapter is [household.py](../cross_episode_sim/env/household.py).

## Atomic training uses datagen too

[episode_setup.py](../cross_episode_sim/generation/episode_setup.py) provides the
same `initialize_environment(config, specification, output, factory)` path for
atomic and composite generation. It configures once, verifies setup, saves initial
state and closes resources if setup fails.

- `prepare_composite_episode(...)` returns the environment, policy, evaluator and setup evidence.
- `execute_atomic_example(config, specification, output, action, factory)` uses the
  same setup path and executes exactly one public atomic call. It returns the live
  environment for caller-managed rendering/closing. Its result is the atomic
  postcondition, not a composite goal.

For future atomic datasets, vary the scene specification/seed in datagen. Do not
add scene constructors or object spawning inside `pick`, `place`, or fixture skills.
The factory is injectable for adapters and tests; injecting a test double does not
establish physical success. Existing standalone fixture demos remain separate
qualification evidence and are not concatenated into composites.

## Timing contract

`scene.minimum_heating_seconds` is 30 seconds by default for the pot and microwave
tasks. The prompt explicitly states the minimum. There is **no heat model, no
public ready signal and no wait-until-ready action**.

The oracle plans use ordinary `wait(duration_seconds=30)` calls. Other planners can
choose different waits or do useful work while an appliance runs. Navigation,
manipulation and waiting all advance simulated time.

The private evaluator counts each pot's time only while it is supported on its
assigned active burner and not held. Microwave time requires the container inside,
the door closed and the appliance active. Removal or switching off pauses counting;
qualifying time accumulates across intervals. Both pots must reach the minimum,
remain on separate assigned burners, and finish with burners off and an empty hand.

[heating_duration.py](../cross_episode_sim/evaluation/heating_duration.py) receives
physics-step boundary observations, not only skill-completion events. Its state
must be restored with simulator checkpoints so rejected oracle attempts do not
contribute time. Duration totals are returned only by `evaluation_snapshot()`, not
by normal observations. Both burners and microwave controls have physical bindings;
complete task rollouts must still pass the private evaluator.

Atomic knob/button tests check contact and the resulting fixture state; they do not
wait 30 seconds. The generic `wait` skill is tested independently.

## Seven independent variants per task

1. Original scene and procedure.
2. Alternative procedure with the same initial scene and goal.
3. Different native kitchen layout.
4. Translated furniture.
5. Rotated living-room furniture (the working tabletop stays axis-aligned).
6. Different object instances within the same categories.
7. Combined variations and alternative procedure.

The 35 configurations are in [configs/composite_pilot](../configs/composite_pilot).
They require no prior successful episode or selected history videos. The pilot is
`pilot_unassigned`, not an established leakage-free train/test split. Matching scene
hashes establish identical specifications, not validated identical physical states.

## Commands

```bash
# Preview prompt, scene specification and atomic plan without simulation.
python -m cross_episode_sim.generation.composite_pilot \
  --describe --task warm_two_pots --variant alternative_procedure

# Preview all 35 variants.
python -m cross_episode_sim.generation.composite_pilot --describe > /tmp/composite_plans.json

# Check installed prerequisites before allocating a GPU.
bash scripts/generate_composite_pilot.sh --preflight

# Run one independent baseline, with an automatically selected free GPU.
bash scripts/generate_composite_pilot.sh --task set_table --variant original

# All 35 configurations, with three independent workers.
bash scripts/generate_composite_pilot.sh --workers 3
```

Use the simulation Python environment, or set `PYTHON=/path/to/sim/bin/python`.
`scripts/generate_composite_pilot.sh` uses independent worker processes, dynamic GPU
leases, per-episode timeouts, a tqdm bar and live `[step/total]` stage changes.
Omitting `--output` creates a unique directory. A failed atomic stops its episode;
the batch records the failure and continues other configurations. Timeout kills
the worker process group, including its scene-authoring children.

`CES_ROBOCASA_PYTHON` selects a separate RoboCasa interpreter; by default the runner
checks the current environment and a sibling `robocasa` virtual environment.
Native scene authoring is CPU-only. Physics/planning/rendering use the leased GPU.

Each episode keeps its config, prompt, oracle plan, task manifest, measured initial
snapshot, `initial_state.npz`, compiled `model.mjb`, complete `skill_calls.json`,
`run.log`, and `report.json`. Successful episodes also contain `composite.mp4` and
`head_camera.mp4` at normal speed. `scene.xml` is the authored house; `model.mjb`
is the actual compiled model including the robot and room doors, suitable for
restoring `initial_state.npz`. Reports preserve failures rather than treating a
partial rollout or a video as task success.

## Validation and remaining physical work

```bash
pytest -q tests/test_composite_tasks.py tests/test_composite_pilot.py \
  tests/test_episode_setup.py tests/test_heating_duration.py
```

Tests cover all 35 recipes, atomic schemas, access/single-gripper constraints,
external goal checks, datagen ownership of setup, and private duration accounting.
Additional regression tests cover box-face grasp contacts, top-down asset selection,
burner surface binding, faithful compiled-model snapshots, and worker timeouts.
See [physical qualification results](composite_pilot_validation.md) for completed
rollouts and remaining failures. The seven variants are not all physically qualified.
