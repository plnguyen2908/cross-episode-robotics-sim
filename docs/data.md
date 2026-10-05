# Data

Tasks read three kinds of files that are not source code: RoboCasa's kitchen assets,
MolmoSpaces' robot and object assets, and this project's **data bundle**. The first two are
downloaded by their own projects at pinned versions (see [installation.md](installation.md)).
The bundle is installed with `scripts/install_data.py` into `data/` (or `CES_DATA_DIR`).

## Bundle contents

| Directory | Contents |
|---|---|
| `grasp_sweep/` | `current_results.json` (outcome of the grasp-qualification sweep for 1,184 objects) and, per object, `initial_scene/` (the three-room house with that object, as authored in RoboCasa) and `status.json` |
| `molmo_objects/` | 688 MolmoSpaces objects converted to standalone MJCF (`model.xml`, `metadata.json`), their DROID grasp library (`grasps.npz`) and TidyBot grasp registry |
| `robocasa_grasps/` | TidyBot grasp registries for RoboCasa objects, mirroring each object's path inside RoboCasa's `models/assets` |
| `base_episodes/breakfast_three_room/` | prepared breakfast scene: house, bound objects and their qualified grasps; the breakfast configs start from it |
| `base_episodes/coffee_counter/`, `coffee_prepare/`, `coffee_tune/` | prepared kitchen state and grasp files the espresso workflow starts from |
| `coffee/espresso_machine/` | espresso machine converted from MoonlakeAI's USDZ, with its Apache-2.0 license and NOTICE |
| `skill_scenes/` | starting scenes for the atomic-skill demonstrations |

### Grasp registry

A grasp enters the registry only after a physical trial: the robot grasps the object in its
starting scene, lifts it and holds it for two seconds with bilateral finger contact. Each entry
(`grasp_annotations_franka_tidybot.json`) stores the object-to-gripper transform, the grasp family
(top, oblique, side), object scale, the trial report it came from, and `model_sha256`: the
SHA-256 of the object's `model.xml` with machine-specific paths replaced by placeholders
(`cross_episode_sim.paths.model_digest`). A grasp is used only for the exact model and scale it
was qualified on, so changing an asset silently disqualifies its grasps rather than reusing
evidence that no longer applies.

### Path placeholders

Scene and model files reference meshes and textures by absolute path. In the bundle those roots
are stored as placeholders, which `install_data.py` resolves on your machine:

| Placeholder | Resolves to |
|---|---|
| `${ROBOCASA_DIR}` | RoboCasa package directory (`ROBOCASA_DIR`) |
| `${MLSPACES_CACHE_DIR}` | MolmoSpaces resource cache |
| `${MLSPACES_ASSETS_DIR}` | MolmoSpaces asset installation |
| `${CES_ASSETS}` | `cross_episode_sim/assets` in this repository |
| `${CES_DATA}` | the installed bundle |
| `${CES_EVIDENCE}` | development-run evidence cited by registries and reports; not shipped |

## Regenerating the data

The bundle was produced by the pipeline below; `scripts/build_data_bundle.py` packages its
outputs. Regenerating it is only needed for new objects, a different house or different assets.

1. **Starting scenes** (environment `robocasa`): `python -m cross_episode_sim.grasping.prepare_recording
   --source {molmo,robocasa} --asset <asset> --output <dir>` builds the three-room house with
   one object placed on the dining table and saves `scene.xml`, `states.npz` and `setup.json`.
2. **Grasp qualification** (environment `sim`, GPU): `python -m cross_episode_sim.grasping.registry_sweep
   --output <sweep_dir> --robosuite-python ~/.venvs/robocasa/bin/python --gpus 0,1` runs up to five
   physical trials per object (top, oblique and side families), creates missing starting scenes
   through step 1, and appends successful grasps to the registries. The full sweep over 1,184
   objects takes several GPU-days.
3. **Breakfast base episode**: `python -m cross_episode_sim.tasks.breakfast.scene --output <dir>`
   binds roles to qualified objects, places them and settles the scene.
4. **Coffee asset**: `pip install -e ".[coffee-asset]"` then
   `python -m cross_episode_sim.tasks.coffee.convert_asset --output data/coffee/espresso_machine`.
5. **Package**: `python scripts/build_data_bundle.py --workspace <dir> --robocasa $ROBOCASA_DIR
   --output dist/data`, then `tar -czf ces-data-v1.tar.gz -C dist data`.
