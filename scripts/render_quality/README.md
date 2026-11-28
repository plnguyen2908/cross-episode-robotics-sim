# Presentation renders (Blender Cycles)

Demo-only rendering of saved episodes. Poses come from the recorded `qpos`
(`action_recording/actions.npz`), replayed kinematically; nothing is simulated or
changed in the episode. These images are not robot observations (AGENTS.md: the
navigation and wrist RGB-D cameras remain the only observation inputs).

```bash
source eval_env.sh
# 1. MuJoCo: default and enhanced renders, plus scene.npz + textures/ for Blender
.venv-sift/bin/python scripts/render_quality/mujoco_frame.py <episode> --frame 37500 --out runs/render_closeup \
  --azimuth 150 --elevation -22 --distance 1.5
# 2. Blender Cycles (venv: uv venv --python 3.11 .venv-render && uv pip install --python .venv-render/bin/python bpy==4.2.0)
.venv-render/bin/python scripts/render_quality/blender_render.py runs/render_closeup --samples 512 \
  --swap bowl_=runs/simready_assets/kitchenware_bowl_c01 \
  --swap cup_=runs/simready_assets/kitchenware_mug_b01 \
  --swap dining_table_,table_leg_=runs/simready_assets/desk_a01 \
  --swap chair_=runs/polyhaven_assets/dining_chair_02
```

`--swap` replaces simulated objects with nicer meshes **in the render only** (see
`asset_swap.py`). Physics, grasps and scoring used the original objects; say so
when showing such a render. `<out>_swaps.json` records the pose fit of each swap.

## Asset sources and licenses

| Source | License | Use here |
|---|---|---|
| [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) | CC BY 4.0: credit NVIDIA | bowls, mugs, food, cans, pans, baskets; `desk_a01` (no dining table or dining chair) |
| [Poly Haven models](https://polyhaven.com/models) | CC0 | dining chairs, wooden tables (glTF) |

Download a SimReady asset (one folder each):
`HF_HUB_OFFLINE=0 .venv-sift/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/simready-assets', repo_type='dataset', local_dir='runs/simready_assets', allow_patterns=['desk_a01/**'])"`

Poly Haven glTF files: `https://api.polyhaven.com/files/<id>` → `gltf` → `2k`
(the `.gltf`, its `.bin` and `textures/`).
