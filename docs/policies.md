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

Currently registered tasks: `breakfast` and `tidy_up`. Coffee runs as a demonstration only: its machine model
still reads the demonstrator's internal phase flags.

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

## Where cross-episode memory fits

The intended setup keeps the low-level policy fixed and adds memory at the planning level:
retrieval of earlier episodes in the same house informs which subtask to do next, with which
parameters and from which base pose. The demonstrator's composite episodes already expose that
level. Each `CompositeEpisode` step is a named operation with arguments, and `report.json`
records every operation's arguments, the candidates tried and the one accepted, so a memory
module can be trained or evaluated on the same structure the demonstrations use.
