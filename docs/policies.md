# Running and training policies

The tasks double as an evaluation environment for learned policies. A policy controls the same
robot the demonstrator uses, in the same scene, and is judged by the same measured goals.

## The environment

```python
from cross_episode_sim.env.task_env import TaskEnv

env = TaskEnv("breakfast", output_root="runs/env", cameras=("robot_0/wrist_cam",), image_size=(224, 224))
observation, info = env.reset(seed=17)
while True:
    action = policy.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(info["goals"], info.get("failure"))
```

**Control rate.** One `step` holds the action for 0.04 s of physics (25 Hz), the rate
demonstrations are recorded at.

**Action**, an 11-dimensional float vector of absolute targets:

| Index | Meaning | Units |
|---|---|---|
| 0–2 | base pose target x, y, yaw in the world frame | m, m, rad |
| 3–9 | FR3 joint position targets, joints 1–7 | rad |
| 10 | gripper command, 0 open, 1 closed | — |

The base and arm are position servos, so targets should move smoothly; a target far from the
current pose produces a fast, possibly unsafe motion, exactly as on a position-controlled robot.

**Observation**, a dict:

| Key | Contents |
|---|---|
| `state` | base x, y, yaw; 7 arm joint positions; current gripper command (11 floats) |
| `<camera name>` | `uint8` image of shape `(H, W, 3)` for each requested camera |

The robot's own camera is `robot_0/wrist_cam`. Any camera defined in the scene can be requested.

**Termination.** An episode ends when every goal predicate holds (reward 1), when physics stops
it, or after `max_steps` (truncation). Physics stops on unsafe events such as robot
self-collision beyond 0.5 mm; `info["failure"]` then holds the reason.

**Goals and events** are defined per task in `cross_episode_sim/env/tasks.py`. For breakfast the
goals are: each vessel on the desk at its setting and upright with its food inside, the fill event
done, storage closed and no vessel in the gripper. The person fills the vessels as soon as all
four stand on kitchen filling surfaces and none is in the gripper, after which the serving
phase begins.

Registered low-level tasks: `breakfast` and `tidy_up`. Coffee and breakfast-with-coffee also support
model planning through the oracle skill interface below. Their machine model still reads internal
demonstrator phase flags, so VLA/direct coffee evaluation is not yet supported.

## Evaluating VLM planners

The current evaluation priority is **VLM planner + oracle skills** and **VLM planner +
VLA skills**. Both use the same public skill catalog and planner decisions. Agent-as-policy
configuration is deferred and will be chosen case by case; its existing hooks are optional.

Run `python -m scripts.evaluate_models` from the repository root. Choose the planning
style independently of the execution backend:

| Planner | Oracle skills | VLA skills | Direct robot actions |
|---|---|---|---|
| `vlm` | Select atomic skill + arguments | Select skill; VLA emits actions | Emit action chunks |
| `agent` | Agent programs call oracle tools | Agent programs call VLA tools | Agent programs call action tools |

An agent controls its own observation/tool loop, including code execution inside your adapter.
This follows the tool-oriented approach of [Agent-as-Policy](https://agent-as-policy-2026.github.io/).
The repository supplies the robot interface, not a hosted coding-agent runtime or trained model.
Adapters configure their provider, checkpoint, inference settings, and request timeouts.

**Supported worlds:** `--task breakfast` supports all six combinations. `--task tidy_up`
supports direct actions only. `cross_episode_sim.evaluation.coffee:from_example` recreates a
selected generated coffee/current episode for either planner with **oracle skills only**.
It rejects VLA/direct execution because those tasks still need independent machine events
and scoring. `--history-example` supplies public inputs; it does not select the current scene.
Use the example session factory to evaluate that exact current layout.

### Batch generation and Qwen evaluation

Generate 30 successful randomized examples, starting at seed 20 and advancing through
new seeds until the target is reached. Each seed’s three episodes run concurrently on the listed GPUs. Run these commands from the repository root:

```bash
python -m scripts.generate_examples --start-seed 20 --count 30 --gpus 0 1 2
```

The launcher prints `BATCH=...`, saves `batch.json` after each attempt, and keeps per-seed
logs and unique attempt directories. The tqdm bar counts validated successes, with the current
seed and failed-seed count alongside it. Failed seeds do not count toward `--count`; the launcher
keeps trying new seeds until all 30 examples pass the same validation used by collection.
Batch workers use `--fail-fast`: a finalized physics failure or nonzero episode exit
stops the seed's remaining workers and diagnostic rendering, then advances to the next
seed. Failure detection polls every second independently of the progress-print interval.
Existing logs, traces already saved, and `generation_failure.json` are retained;
videos from failed seeds may be incomplete. Individual examples can omit `--fail-fast`
when complete diagnostic videos are wanted. Restart an existing batch launcher with
`--resume` to enable this behavior.
Resume preserves the saved success target and scene settings; an already-complete batch does
no further work. Legacy batches use their original seed count as the success target.
`--retry-failed` retries each failed seed at most once per invocation before moving to new seeds.
Base speeds are explicit in new batch settings: `--nav-speed` (m/s) and
`--turn-speed` (rad/s). Defaults are 0.24 and 0.6; coffee previously inherited
0.08 rad/s turning from its saved preparation configuration. Explicit speed flags
on resume override speeds for subsequent attempts, not already-running workers.
The defaults match `--nav-speed .24 --turn-speed .6`; reverse undocking remains
capped at .08 m/s and the existing collision, grip, and tracking checks remain active.
Higher speeds need physical validation on new layouts and are not a full-batch runtime guarantee.

Coffee and current scene setup reuse asset exports within each episode and reuse compiled
geometry when only a free object's placement changes. Physical settling and support checks
still run. Each episode writes `setup_timings.json`; the progress display names the active
setup stage, and its worker log records completed-stage times. Changes apply to newly
started workers, including subsequent seeds of an existing batch.
The current oracle witness serves each cup directly from the coffee machine to its
dining setting, then fills and serves the bowls. It omits the intermediate kitchen
return and second cup pickup, saving four public skill calls per cup. Coffee history
still returns brewed cups to their homes. Every action uses the same public skill API.
Successful spout extraction resets the machine for its next brew while preserving
power and the list of filled cups, so direct serving supports multiple cups too.

Spout placement approaches above the seated cup height, checks and corrects the
measured cup pose between short insertion segments, then lowers the centered cup
onto the tray. The public `place` skill exposes bounded internal clearance trials
(1, 0.5, and 1.5 mm); the oracle can explicitly redock through `navigate` after
tracking failures exhaust those trials. Collision and payload-pose tolerances
remain unchanged. Tracking failures record their stage, target/actual TCP positions,
joint-command errors, and contact pairs in the rejected-trial diagnostics.

Seed-20 preparation profiling (three settings, current with six cups and six bowls)
measured current setup at 393 s before reuse and 246 s afterward, with scene
compilations reduced from 170 to 66. Optimized coffee preparation measured 113 s.
These times exclude Python imports, controller initialization, execution, and rendering;
they are single-run measurements on a shared host. Both current scenes selected the same
objects; task-body poses differed by less than 0.1 micrometre, with small settling
differences in clutter (under 0.05 mm translation and 0.33 degrees rotation).

Resume using the printed directory:

```bash
python -m scripts.generate_examples --resume runs/generation_batch_YOUR_ID --gpus 0 1 2
# Also retry failed seeds, retaining failed attempts:
python -m scripts.generate_examples --resume runs/generation_batch_YOUR_ID --retry-failed --gpus 0 1 2
```

Collect all complete successes, including existing runs and newly generated batches:

```bash
python -m scripts.collect_examples --source runs --output datasets/successful_examples
```

Collection requires three successful episodes, nonempty passing goal evidence, completed
renders, and readable first/last video frames. It skips incomplete/failed examples, records
reasons in `rejected.json`, and is safe to repeat as generation progresses. Sources are
preserved; videos are hardlinked when possible, copied across filesystems. Treat the resulting
dataset files as immutable because hardlinks share file contents with their source.
`dataset.json` lists every accepted run, its source, seed and layout fingerprint. Repeated
layouts are retained; `--unique-layouts` during evaluation avoids giving them extra weight.

The Qwen adapter defaults to **Qwen/Qwen3.5-9B**, a native vision-language model. It calls a
chat-completions endpoint, so inference can run on another GPU or host. For a local server,
use a separate environment with PyTorch, Pillow, Transformers and Accelerate:

```bash
pip install 'transformers>=5.2,<6' accelerate pillow
CUDA_VISIBLE_DEVICES=7 python -m scripts.serve_qwen --model Qwen/Qwen3.5-9B --port 8019
```

The local helper uses BF16, SDPA, deterministic generation, and disables thinking output.
It binds to localhost and accepts inline camera/history images. The optional `--checkpoint`
points to downloaded weights. Tested library versions and model revision are recorded in the
validation artifacts; other compatible endpoints can use the same adapter. For authenticated
endpoints set `QWEN_API_KEY` in the caller's environment.

Start with a short test in the simulator environment, using a different GPU:

```bash
python -m scripts.evaluate_dataset --dataset datasets/successful_examples \
    --endpoint http://127.0.0.1:8019/v1 --gpu 4 \
    --limit 1 --history-frames 8 --max-planner-steps 3
```

Then omit `--max-planner-steps` for full tasks and use `--unique-layouts` to evaluate one
example per distinct layout. `--ids ID ...` selects particular dataset entries. Outputs
use a new `runs/qwen_oracle_evaluation_*` directory; `summary.json` distinguishes limited
skill tests from full task evaluation and reports both oracle-call outcomes and task success.
A budget stop on an unfinished task remains `success=false`.

The collector freezes each episode's authored initial scene XML and oracle grasp annotations
in `initial_scenes/`, separate from the model's inputs. The evaluator loads that initial scene
instead of resampling clutter from a seed. Shared external meshes/textures still require the
normal repository data installation. It also checks recorded object, furniture, serving, and
machine layouts. A layout mismatch stops that trial with an error rather than silently
evaluating a different arrangement. No successful rollout checkpoint is used to initialize it.
Symbolic planner adapters receive complete history-video paths and decide how to decode,
sample, or otherwise process them. The Qwen adapter currently chooses uniform sampling
(default eight frames per video, configured through its `history_frames` model option or
`evaluate_dataset --history-frames`). This is an adapter choice, not a harness restriction.
Qwen sees the public instruction, its sampled history frames, current wrist-camera images,
skill descriptions/schemas, and previous calls/feedback. Symbolic observations contain only
`images`: no base pose, arm joints, gripper command, calibration, or simulation time.
Oracle execution and VLA adapters retain their robot-sensor observations.
It chooses each next atomic call itself. Neither `current.mp4`, the oracle witness plan nor
private simulator reports are sent to the model. The adapter also supports the VLA backend
through `scripts.evaluate_models`; dataset coffee/current evaluation currently supports
oracle execution only.

### Adapter contracts

A VLM planner implements `reset(context)` and `plan(observation, feedback)`:

```python
class MyPlanner:
    def reset(self, context):
        # instruction, histories, seed, skill_catalog, action_spec, budgets, skill_backend
        self.context = context

    def plan(self, observation, feedback):
        # Return exactly one of these forms:
        return {"skill": "pick", "arguments": {"role": "cup_one", "source": "surface"}}
        # {"actions": [[...11 absolute targets...], ...]}  # direct backend only
        # {"finish": True}
```

Use names and argument schemas from `context["skill_catalog"]`; domains differ by task.
`context["planner_decision_schema"]` supplies a JSON Schema for structured model output:
each skill name is paired with its own argument schema, plus `{"finish": true}`.
Oracle and VLA backends expose the same decision schema. Provider adapters can translate
it to their provider's supported structured-output format; runtime validation always checks
the actual call. No oracle preconditions or reference action sequence enter this schema.
Observations contain `state` (11 floats), `images` (camera name → RGB uint8 array), and
`sim_time`. Feedback reports skill status and an error category, without oracle poses,
private preconditions, or final goal predicates. A failed oracle skill does not automatically
redock or retry at the evaluation layer: the model can explicitly choose another call.
Oracle skills retain their internal motion-search and rollback privileges; results mark this
with `oracle_rollback_trials=true`, so their action counts are not comparable with VLA counts.

A skill-conditioned VLA implements:

```python
class MyVLA:
    def start_skill(self, name, arguments, context):
        # Reset skill-local state and condition on the requested skill.
        ...

    def act(self, observation):
        return {"actions": [[...11 absolute targets...], ...]}
        # Or {"done": True} to return control to the planner.
```

The VLA receives fresh observations between chunks. Actions are the absolute base/joint/gripper
targets documented above, at 25 Hz. Maximum chunk length defaults to 64; each action consumes
the episode budget. The VLA backend never executes oracle skill motion. Its `done` signal is
reported as `model_reported_done` with `verification=unverified`; only the independent task
verifier can award episode success. The per-skill limit returns `skill_budget` to the planner;
an episode limit stops evaluation.

An agent adapter implements `run(context, tools)` and exposes these methods to its model or
coding runtime:

```python
observation = tools.observe()
feedback = tools.execute_skill("pick", {"role": "cup_one", "source": "surface"})  # oracle or VLA
feedback = tools.act([[...11 targets...]])                    # direct backend
status = tools.status()                                      # budgets + feedback
result = tools.finish()                                      # completion claim only
```

These are trusted Python adapters, not a security sandbox. Do not give model-generated code
filesystem access to private episode artifacts, controller internals, or evaluator state.
`finish()` and agent return do not establish success. The evaluator checks task goals.

### Agent control below atomic skills

With `--planner agent --backend direct`, the agent may choose task actions itself and call:

| Tool | Arguments | Execution |
|---|---|---|
| `move_joints` | 7 joint angles in radians, `duration` | Interpolate targets; joint position servos track them |
| `move_base` | world `[x_m, y_m, yaw_rad]`, `duration` | Interpolate mobile-base position targets |
| `gripper` | normalized `value` (0 open, 1 closed), `duration` | Drive the gripper actuator |
| `move_ee` | world position in meters, unit quaternion `[w,x,y,z]`, `duration` | Local numerical IK, then joint interpolation |
| `kinematics` | optional `joints` and world `base` | Robot TCP transform, world-frame 6×7 Jacobian, joint limits |
| `act` | chunks of 11-dimensional targets | Direct control intervals, without interpolation |

This supports both controller-provided IK (`move_ee`) and agent-written IK: query
`kinematics`, compute joint updates using its Jacobian or another numerical method, then call
`move_joints` or `act`. FK/IK queries use scratch robot state, do not move the live robot, and
provide no object poses, grasp choices, or task-aware planning. `move_ee` keeps the base fixed;
its tolerance is 2 mm / 1 degree. IK failure makes no motion. A reachable target does not imply
a collision-free path; execution retains the environment's physical checks.

Motion tools allow durations up to 10 simulated seconds. Every generated control interval
consumes the same action/time budgets as `act`. Feedback includes final measured state and
target error; `executed` means commands ran, not that the task succeeded. Observations also
include camera intrinsics and `world_from_optical` transforms (optical x right, y down, z forward),
so the agent can calculate visual targets without access to simulator object coordinates.
No-op physical smoke test for these tools:

```bash
python -m scripts.evaluate_models --task breakfast --planner agent --backend direct \
    --model examples.evaluation_plugins:RobotToolsAgent --gpu 0
```

For an example that computes its own Jacobian-based IK and executes a 5 mm motion,
use `--model examples.evaluation_plugins:ProgrammedIKAgent` with the same agent/direct flags.
Its `agent_written_position_ik` function uses only public kinematics queries and
`move_joints`; it does not call the supplied `move_ee` solver. These are interface checks,
not trained agent task-success demonstrations.

### Commands and outputs

A quick VLM/VLA interface smoke test (holds position; **expected task success is false**):

```bash
python -m scripts.evaluate_models --task breakfast --planner vlm --backend vla \
    --model examples.evaluation_plugins:OneSkillPlanner \
    --model-options '{"skill":"navigate_to_wait","arguments":{}}' \
    --vla examples.evaluation_plugins:HoldVLA --gpu 0
```

Your VLM with oracle skills, and then with learned skills:

```bash
python -m scripts.evaluate_models --task breakfast --planner vlm --backend oracle \
    --model my_models:Planner --model-options '{"checkpoint":"path/to/planner"}' \
    --episodes 10 --seed 100 --gpu 0

python -m scripts.evaluate_models --task breakfast --planner vlm --backend vla \
    --model my_models:Planner --model-options '{"checkpoint":"path/to/planner"}' \
    --vla my_models:SkillVLA --vla-options '{"checkpoint":"path/to/vla"}' \
    --episodes 10 --seed 100 --gpu 1
```

For an agent use `--planner agent --model my_models:Agent` with either backend.
For a direct-action policy use `--backend direct` (no `--vla`).
`examples.evaluation_plugins:HoldPlanner` with `--planner vlm --backend direct`
is a no-op action smoke test, not a baseline policy.

A VLM planner with oracle skills in the **exact current episode** of a generated example:

```bash
python -m scripts.evaluate_models --planner vlm --backend oracle \
    --model my_models:Planner --session cross_episode_sim.evaluation.coffee:from_example \
    --session-options '{"example":"runs/YOUR_EXAMPLE"}' \
    --history-example runs/YOUR_EXAMPLE --gpu 2
```

The evaluator privately reads the example manifest to rebuild the scene. The model receives
only `prompt.txt` and complete `history_1.mp4` and `history_2.mp4` paths in
`context["histories"]` (`name`, `video_path`, `mime_type`). Each planner adapter controls
its video processing; the harness does not sample or truncate the videos. It receives neither `current.mp4`,
the witness plan, nor private reports. The example fixes the scene seed; `--seed` becomes
model-trial context. Without an example session, seeds select newly generated task scenes.

Omitting `--output` creates a unique `runs/model_evaluation_*` directory. If an explicit output
already exists, the launcher adds a unique suffix and preserves the earlier run. Every trial writes `episode_NNN/result.json`, `events.jsonl`, and numbered
observation `.npz` files; `summary.json` records all outcomes and success rate. Setup errors
also get failed results. Model and VLA instances are recreated per trial.
Sensor archives store state and RGB images; observation events map image indices to cameras.

Default limits: 256 public tool calls (including observations and invalid requests), 25,000
low-level actions, 2,500 actions per VLA skill, 1,800 simulated seconds, 3,600 wall seconds.
Override with e.g. `--budgets '{"wall_seconds":7200,"actions":50000}'`. Wall time begins
after scene setup/history loading, includes inference, and is checked between calls and after
execution. It cannot interrupt an in-flight provider request or oracle skill; configure provider
timeouts in adapters. Over-budget completions fail. Oracle action counts exclude internal
controller steps; use wall/simulation time and logged skill calls instead.

For custom domains, wrap an initialized public interface with
`evaluation.sessions.ControllerSession(controller, interface, verify_goal)`. It defaults to
oracle-only. Low-level support requires an explicit action adapter implementing `step(action)`
and independent `score()` predicates. A CLI session factory accepts `output_root`, `seed`,
and your `--session-options`, and returns a session implementing `observe`, `catalog`,
`validate_skill`, `oracle_skill`, `step`, `score`, `sim_time`, `action_spec`, and `close`,
plus `instruction` and `supports_actions`. Scores must be a nonempty dictionary of boolean goals.

## Rolling out a policy

Any object with `act(observation) -> action` (and optionally `reset()`) can be evaluated:

```bash
python scripts/run_policy.py --task breakfast --policy my_package.policies:MyPolicy \
    --policy-arg checkpoint=path/to/ckpt --episodes 10 --seed 100 --output runs/eval
```

Each episode directory gets `rollout.mp4` and `result.json` (steps, success, goals, failure);
`summary.json` holds the success rate.

## Building a training set

1. **Generate demonstrations.** Run the task over many seeds; each successful run is one demo.

   ```bash
   for seed in $(seq 100 199); do
     python -m cross_episode_sim.tasks.breakfast.gather --seed $seed --output runs/demos/breakfast_$seed
   done
   ```

   Runs are independent and CPU-bound apart from cuRobo, so run several in parallel per GPU.

2. **Export** to robomimic-style HDF5. Images are rendered by replaying the recorded states, so
   choose cameras and resolution at export time:

   ```bash
   python -m cross_episode_sim.data.export runs/demos/breakfast_* --output datasets/breakfast.hdf5 \
       --camera robot_0/wrist_cam --image-size 224 224 --stride 1
   ```

   Unsuccessful runs are skipped unless `--include-failures` is given. Each demo stores
   `actions`, `obs/state`, `obs/<camera>_image`, the full simulator `states` for exact replay,
   and attributes `success`, `task`, `instruction` and the source run. Actions are the commands
   the controller sent; `--stride N` subsamples to 25/N Hz.

3. **Train.** `examples/train_bc.py` is a minimal behavior cloning template (small CNN on the
   wrist image plus state, MSE on normalized actions):

   ```bash
   python examples/train_bc.py --dataset datasets/breakfast.hdf5 --output checkpoints/bc.pt --epochs 20
   python scripts/run_policy.py --task breakfast --policy examples.train_bc:BCPolicy \
       --policy-arg checkpoint=checkpoints/bc.pt --output runs/bc_eval
   ```

   It demonstrates the data path, not a competitive baseline. The HDF5 layout follows
   robomimic's (`data/demo_i/{actions,obs,states}`), which is the natural starting point for
   training with robomimic or other HDF5-based pipelines.

4. **Check action semantics.** `examples/replay_policy.py` replays a demo's actions open-loop.
   Replaying a successful demonstration in its own episode should track the recorded states:

   ```bash
   python scripts/run_policy.py --task breakfast --seed 17 --policy examples.replay_policy:ReplayPolicy \
       --policy-arg dataset=datasets/breakfast.hdf5 --output runs/replay_check
   ```

## Using past episodes

The goal of the project is to finish the current episode using experience from past ones,
retrieved by similarity or taken in full. Keep the low-level policy fixed and let past episodes
inform the planning level: which subtask to do next, with which arguments, from which base pose,
and which candidates to skip because they failed before.

The recorded episodes expose exactly that level. In every run directory:

- `instruction.txt` and `task_manifest.json` describe the task and the scene it ran in;
- `composite_execution.json` lists the plan as named operations with arguments, with timing and
  outcome for each;
- `report.json` holds every stage's measurements, the candidates each skill tried, the one it
  accepted, and the measured goal evidence;
- `rejected_trials/` keeps the failed attempts with their reasons;
- `trace.json` is the full state trajectory, exportable as training data.

To evaluate, run the same new episode with and without access to the past ones and compare
success, time to completion and the number of failed attempts.

### History views and surplus vessels

New cross-episode videos use a synchronized 2x2 layout: kitchen, dining, living,
and the robot first-person camera (wrist-mounted on Franka). Room cameras are
fixed perspective cameras at high wall corners, 2.6 m above the floor and
20 cm off the side wall, with a 75° vertical field of view and aimed toward room center at 1.1 m height. The kitchen
mount is 90 cm along the wall from the corner to see past the doorway jamb;
the other mounts are 20 cm from both corner walls. Walls and
furniture occlude them normally. Only panel
names are overlaid; controller stage text and simulator-state banners are omitted.
Existing videos are unchanged and must be regenerated to use this layout.

New full-breakfast examples default to six cups and six bowls in the current
episode, independent of the requested 1–3 place settings (`--extra-vessels 0` restores the earlier exact-count scene; a positive value
sets the pool size to three plus that value per type).
All pool vessels are callable through the same skill interface, with shuffled IDs.
Histories retain the requested serving count. Current success requires exactly that
number of filled cups and bowls supported on the dining table, paired one-to-one;
any pool vessels may satisfy it. `navigate(..., destination="serving", setting="1")`
selects a dining setting; `setting` accepts the generic maximum range 1–3 without
revealing the requested count. Unused vessels stay outside the dining table.

The Franka panel renders the actual model camera `robot_0/wrist_cam`, attached
to `wrist_cam_body` under `fr3_link7`. The upstream `franka_droid/model.xml` places
that body at `[0.05, 0, 0.107]` m relative to link 7 and uses `fovy=90`. Its source
contains a TODO to add the camera mount and move the camera. This matches the
simulator asset; it is not a calibrated physical-camera specification.
