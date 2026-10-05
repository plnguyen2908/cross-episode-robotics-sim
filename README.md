# Cross-Episode Robotics Sim

Multi-room household tasks for a mobile manipulator in [RoboCasa](https://robocasa.ai), built
for studying robots that get better at the same task each time they repeat it.

A TidyBot++ holonomic base carries a Franka FR3 arm with a Robotiq 2F-85 gripper through a
kitchen, dining room and living room. Everything is physically simulated in MuJoCo: objects move
only through gripper contact, food and coffee grounds are free rigid bodies, and doors, drawers,
knobs and buttons move on their native joints.

The repository provides

- **two composite tasks** with oracle demonstrations: breakfast for two (gather empty vessels
  from three rooms, a person fills them, serve them) and making espresso;
- **ten atomic skill families** they are built from: navigation, pick and place, doors, drawers,
  knobs, levers, buttons, insertion, sliding racks and lids, each validated on at least one fixture;
- **a template task** (`tidy_up`: return a mug and a book to their rooms) that shows how tasks are
  composed from the skills;
- **a Gymnasium environment** where a learned policy drives the same robot and is scored by the
  same measured goal predicates;
- **dataset export** from demonstrations to robomimic-style HDF5, with a minimal behavior
  cloning example.

| Breakfast for two | Making espresso |
|---|---|
| [▶ breakfast.mp4](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/breakfast.mp4) · 816 s simulated, shown at 3× | [▶ coffee.mp4](https://github.com/plnguyen2908/cross-episode-robotics-sim/releases/download/data-v1/coffee.mp4) · 330 s simulated, shown at 3× |
| Collect two mugs and two bowls from three rooms, wait while a person fills them with randomly chosen food, then carry each filled vessel to the office desk (the dining-room table). | Twist out the portafilter, pour 32 grounds from a box, reinstall and lock it, put a cup under the spout, press power, then brew. |

## Why

Physical agents today pair a planner with skill policies and start every episode from scratch.
The hypothesis behind this project is that an episodic memory over a fixed base policy can
replace search with recall: which dock works beside a crowded counter, which grasp failed
yesterday, where the mugs were put away. Existing benchmarks evaluate each query once. These
tasks are meant to be repeated over many simulated days in the same house, so improvement
across episodes can be measured.

## Quick start

Two Python 3.11 environments are used, because RoboCasa pins `mujoco==3.3.1` while the
simulator here uses `mujoco==3.5.0`. The RoboCasa environment is only needed to download its
assets and to regenerate starting scenes.

```bash
# RoboCasa at the tested commit, with the three-room house patch (environment: robocasa)
git clone https://github.com/ARISE-Initiative/robosuite && git -C robosuite checkout 5ce6643
git clone https://github.com/robocasa/robocasa && git -C robocasa checkout 4f8a298
git -C robocasa apply ../cross-episode-robotics-sim/third_party/robocasa/three_room_house.patch
pip install -e robosuite -e robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets      # ~10 GB

# This package (environment: sim; needs a CUDA GPU for cuRobo)
cd cross-episode-robotics-sim
pip install -e ".[dev]"
export ROBOCASA_DIR=/path/to/robocasa/robocasa           # the inner package directory
python scripts/install_data.py                           # starting scenes, grasp registry (~500 MB)

# Run demonstrations (headless rendering through EGL)
export MUJOCO_GL=egl
python -m cross_episode_sim.tasks.breakfast.gather --seed 17 --output runs/breakfast
python -m cross_episode_sim.tasks.coffee.workflow --output runs/coffee
scripts/run_skill.sh drawer runs/drawer
```

Each run directory holds `report.json` (measured outcome and every stage), `trace.json`
(simulator state at 25 Hz), the scene, and videos rendered after physics ends.
See [docs/installation.md](docs/installation.md) for details and troubleshooting.

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Environments, RoboCasa patch, MolmoSpaces assets, data bundle |
| [Tasks](docs/tasks.md) | Breakfast, coffee and the atomic skills: what they do, success criteria, commands |
| [Running and training policies](docs/policies.md) | `TaskEnv` API, rollouts, dataset export, behavior cloning example |
| [Adding a task](docs/adding_a_task.md) | Scene preparation, transactional skills, composite episodes, goals |
| [Architecture](docs/architecture.md) | Package layout, controller layers, navigation and manipulation stack |
| [Data](docs/data.md) | What the data bundle contains and how to regenerate it |

## Repository layout

```
cross_episode_sim/
  tasks/breakfast, tasks/coffee   composite tasks: scene preparation, episode controller, CLI
  fixtures/                       atomic fixture skills (doors, drawers, knobs, levers, buttons, ...)
  skills/                         transactional skill runner, checkpoints, composite episodes
  manipulation/                   RoboCasa manipulation: grasp qualification, recovery, placement
  navigation/                     SE(2) route planners and holonomic route execution
  controller/                     shared base controller: scene, physics, grasping, arm motion
  robot/                          TidyBot++/FR3 embodiment, robot profile, cuRobo arm planner
  env/                            Gymnasium environment and task hooks for learned policies
  data/                           dataset export, trajectory cleanup, run provenance
  grasping/                       grasp-qualification sweep that builds the grasp registry
scripts/                          data install, skill launcher, policy rollouts, bundle builder
examples/                         replay policy, behavior cloning
third_party/robocasa/             three-room house patch for RoboCasa
```

## Development

```bash
pytest                 # unit tests: geometry, state logic, skill wiring
ruff check . && ruff format --check .
```

Full demonstrations are the integration tests: run breakfast, coffee and `scripts/run_skill.sh`
for each skill, and check `report.json` for `"success": true`.

## Status and scope

- Demonstrations come from an **oracle controller** that reads simulator state (object poses and
  geometry). They are ground truth for evaluating memory and policies, not a camera-only policy.
- Coffee brewing is a timed state model without fluid or heat simulation. Grounds and food are
  free rigid bodies.
- The policy environment supports **breakfast**. Coffee's machine model still reads the
  demonstrator's internal phase flags, so coffee runs as a demonstration only for now.
- Composite tasks are validated on the seeds listed in [docs/tasks.md](docs/tasks.md); atomic
  skills on the fixture and object listed there, not on every RoboCasa asset.

## Acknowledgments

Built on [RoboCasa](https://github.com/robocasa/robocasa), [RoboSuite](https://github.com/ARISE-Initiative/robosuite),
[MolmoSpaces](https://github.com/allenai/molmospaces) (robot models, object assets, planners),
[cuRobo](https://github.com/NVlabs/curobo) and [MuJoCo](https://github.com/google-deepmind/mujoco).
The TidyBot++ base model comes from [tidybot2](https://github.com/jimmyyhwu/tidybot2) (MIT) and
the espresso machine from [MoonlakeAI sim-env-builder](https://github.com/MoonlakeAI/sim-env-builder)
(Apache-2.0); their license files ship with the assets.

## License

Apache-2.0, see [LICENSE](LICENSE).
