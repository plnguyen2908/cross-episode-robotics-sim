# Installation

Tested on Linux with Python 3.11, NVIDIA RTX 3090 GPUs and CUDA 12. A GPU is required: arm
motion is planned with cuRobo. Simulation itself runs on the CPU, so several episodes can share
one GPU.

## 1. Two environments

RoboCasa pins `mujoco==3.3.1` and `numpy==2.2.5`; this package uses `mujoco==3.5.0`. Keep them
in separate virtual environments (or conda environments):

| Environment | Contains | Used for |
|---|---|---|
| `robocasa` | RoboSuite, RoboCasa (patched) | downloading RoboCasa assets; regenerating starting scenes and the grasp sweep |
| `sim` | this package, MuJoCo 3.5, cuRobo, MolmoSpaces | running tasks, the policy environment, dataset export |

The `sim` environment never imports RoboCasa. It reads RoboCasa's asset files from the directory
named by `ROBOCASA_DIR`.

## 2. RoboCasa with the three-room house

The house used by every task joins a native RoboCasa kitchen to a furnished dining room and
living room. That extension is a patch against RoboCasa commit `4f8a298`. Run this from the
root of this checkout; the clones go into the git-ignored `external/` directory, because clones
directly in the root would shadow the installed `robocasa` and `robosuite` packages on import:

```bash
python3.11 -m venv ~/.venvs/robocasa && source ~/.venvs/robocasa/bin/activate
git clone https://github.com/ARISE-Initiative/robosuite external/robosuite
git -C external/robosuite checkout 5ce6643
git clone https://github.com/robocasa/robocasa external/robocasa
git -C external/robocasa checkout 4f8a298
git -C external/robocasa apply "$PWD/third_party/robocasa/three_room_house.patch"
pip install -e external/robosuite -e external/robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # about 23 GB on disk
deactivate
```

The patch adds `robocasa/models/scenes/dining_room.py`, a custom kitchen–living-room layout,
registry entries and two preview scripts (`python -m robocasa.scripts.preview_kitchen_living_room`).

## 3. This package

```bash
python3.11 -m venv ~/.venvs/sim && source ~/.venvs/sim/bin/activate
pip install -e ".[dev]"
```

This installs MolmoSpaces from GitHub at the pinned commit `7351388`, which supplies the FR3 and
Robotiq models, MolmoSpaces object assets and the A* planner. MolmoSpaces downloads its assets
(about 20 GB) on first use into `~/.cache/molmo-spaces-resources` and `~/.cache/molmospaces/assets`; set
`MLSPACES_CACHE_DIR` and `MLSPACES_ASSETS_DIR` to put them elsewhere. Asset versions are pinned
by MolmoSpaces, so a scene authored on one machine resolves to the same files on another.

## 4. Data bundle

```bash
export ROBOCASA_DIR="$PWD/external/robocasa/robocasa"   # inner package directory, contains models/
python scripts/install_data.py
```

The script downloads `ces-data-v1.tar.gz` from the GitHub release, unpacks it into `data/`
(override with `CES_DATA_DIR`) and rewrites the path placeholders in the bundle to your RoboCasa,
MolmoSpaces and package directories. It prints each placeholder and the directory it resolved
to. Use `--archive` for a downloaded archive or `--source` for an unpacked bundle.
See [data.md](data.md) for the contents.

## 5. Check the installation

```bash
export MUJOCO_GL=egl
pytest                                        # unit tests, no GPU needed
python -m cross_episode_sim.tasks.breakfast.gather --seed 17 --prepare-only --output runs/check
```

`--prepare-only` builds and settles the scene without running the robot. It should print the
selected vessels and, for seed 17, food assets `SugarCube005`, `CookieDoughBall002`,
`IceCube005` and `Potato_21`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ROBOCASA_DIR` | found via `import robocasa` | RoboCasa package directory holding `models/assets` |
| `CES_DATA_DIR` | `<repo>/data` | installed data bundle |
| `CES_RUNS_DIR` | `<repo>/runs` | default output root |
| `MLSPACES_CACHE_DIR`, `MLSPACES_ASSETS_DIR` | MolmoSpaces defaults | MolmoSpaces resource and asset caches |
| `MUJOCO_GL` | — | `egl` for headless rendering |
| `CUDA_VISIBLE_DEVICES` | — | GPU used by cuRobo |

## Troubleshooting

- **`Set ROBOCASA_DIR ...`** — the `sim` environment cannot import RoboCasa by design; point
  `ROBOCASA_DIR` at the patched checkout's inner `robocasa/` directory.
- **`No matching current robot/model/scale evidence`** — an object model differs from the one the
  grasp registry was built for, usually because RoboCasa assets or MolmoSpaces assets are a
  different version. Re-download the pinned versions, or rebuild the registry (see data.md).
- **Black or empty videos** — set `MUJOCO_GL=egl` and make sure the EGL driver is installed.
