#!/usr/bin/env bash
# Reproducible, self-contained environment for the household evaluation
# (scripts/eval_households.py -> scripts/eval_household.sh -> run_*_household.py)
# and its gRPC model services (scripts/serve_sensor_model.py --kind qwen|molmo|fastsam|m2t2|graspgenx).
# Idempotent: rerun after a failure and it continues; nothing outside this checkout
# is written except the uv/pip/HF caches you point it at (see "Environment knobs").
#
#   bash scripts/setup_eval_env.sh            # everything
#   SKIP_HF_MODELS=1 bash scripts/setup_eval_env.sh   # no Qwen/Molmo download (36 GB)
#
# Creates (all inside this checkout, no layering on other users' paths):
#   external/{DAAAM,daaam-boxmot,M2T2,GraspGenX,molmospaces,robosuite,robocasa}  pinned commits + patches/eval_env/*.patch
#   .venv-sift       main runtime + qwen/molmo/fastsam services   (requirements-eval-sift.txt, exact, --no-deps)
#   .venv-m2t2       M2T2 service, pointnet2_ops compiled          (requirements-eval-m2t2.txt)
#   .venv-graspgenx  GraspGen-X service                            (requirements-eval-graspgenx.txt)
#   external/DAAAM/checkpoints/fastsam/FastSAM-s.pt, GraspGenX checkpoints + gripper descriptions,
#   HF snapshots (pinned revisions) of Qwen/Qwen3-VL-8B-Instruct, allenai/Molmo2-4B, wentao-yuan/m2t2
#   eval_env.sh      `source` it before running anything (PYTHONPATH, MUJOCO_GL, HF_HOME, ...)
# and finishes with scripts/check_eval_env.py in each venv (imports + CUDA checks).
#
# ---------------------------------------------------------------- system requirements
# * Linux x86_64, NVIDIA driver supporting CUDA 12.6+ (driver >= 560). No sudo needed.
# * git, curl, a C/C++ compiler (gcc/g++ 9-13; tested with 13.3) for pointnet2_ops and sdists.
# * nvcc 12.x for ONE extension (M2T2 pointnet2_ops). Order of use: $CUDA_HOME (if its nvcc is 12.x),
#   `nvcc` on PATH (e.g. `module load cuda/12.6`), else a minimal CUDA 12.6 compiler is downloaded
#   into .venv-eval-cache/cuda-12.6/ from the conda `nvidia` channel (scripts/fetch_cuda_toolkit.py, ~1 GB).
#   cuRobo needs no nvcc: at this commit it compiles kernels at run time with NVRTC (pip wheels).
# * Headless MuJoCo rendering: libEGL.so.1 from the NVIDIA driver (libnvidia-egl / GLVND,
#   normally installed with the driver on GPU nodes) and libOpenGL/libGL. Set MUJOCO_GL=egl.
#   Open3D additionally loads libGL.so.1, libgomp.so.1 (gcc runtime) and libX11 (from the system).
# * uv (https://docs.astral.sh/uv/) — installed into ~/.local/bin if missing; it downloads
#   CPython 3.11.14 (python-build-standalone), so no system Python is needed.
# * Internet access to github.com, pypi.org, huggingface.co, api.anaconda.org (run on a login
#   node if compute nodes are offline; the result is relocatable only by rerunning this script).
# * Disk: ~25 GB checkout + venvs + caches, +13 GB MolmoSpaces resources, +36 GB Qwen/Molmo HF snapshots.
#
# ---------------------------------------------------------------- environment knobs
#   HF_HOME              HF cache for model snapshots (default: <repo>/.venv-eval-cache/huggingface)
#   UV_CACHE_DIR         uv download cache (uv default ~/.cache/uv)
#   UV_PYTHON_INSTALL_DIR where uv puts CPython 3.11.14 (uv default ~/.local/share/uv/python)
#   CUDA_HOME            CUDA 12.x toolkit to compile pointnet2_ops
#   TORCH_CUDA_ARCH_LIST GPU archs for pointnet2_ops (default "8.0;8.6;8.9;9.0": A100/3090/L40/H100)
#   SKIP_HF_MODELS=1     skip Qwen3-VL/Molmo2 downloads; SKIP_CHECK=1 skip the final self-check
#   SKIP_ROBOCASA=1      do not clone robosuite/robocasa (only needed as asset sources; see below)
#   MLSPACES_CACHE_DIR / MLSPACES_ASSETS_DIR  MolmoSpaces resources (default <repo>/.venv-eval-cache/...);
#   SKIP_MLSPACES_PREFETCH=1 skip their ~13 GB download (it then happens on first scene load)
#
# Not done here (data, not environment): saved episodes reference meshes/textures by absolute
# path; see scripts/pack_eval_episodes.py to copy episodes + exactly the assets they use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY_VERSION=3.11.14
export HF_HOME=${HF_HOME:-$ROOT/.venv-eval-cache/huggingface}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}
log(){ printf '\n== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- tools
for t in git curl gcc g++; do command -v $t >/dev/null || die "$t is required (see header)"; done
if ! command -v uv >/dev/null; then
  log "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/0.9.17/install.sh | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
uv python install --no-bin "$PY_VERSION"   # --no-bin: do not touch ~/.local/bin
PYBIN=$(uv python find "$PY_VERSION")

# ---------------------------------------------------------------- external repos
clone_pinned(){ # name url commit
  local dir="external/$1" url=$2 commit=$3
  if [ -L "$dir" ]; then die "$dir is a symlink; this script needs real checkouts (remove the link in a fresh clone)"; fi
  if [ ! -d "$dir/.git" ]; then
    log "clone $1 @ ${commit:0:8}"
    rm -rf "$dir.partial"
    git clone --filter=blob:none -q "$url" "$dir.partial"
    git -C "$dir.partial" checkout -q --detach "$commit"
    mv "$dir.partial" "$dir"
  fi
  [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || die "$dir is not at $commit; refusing to touch it"
}
apply_patch(){ # dir patch
  local dir=$1 patch=$ROOT/$2
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then echo "patch already applied: $2"
  else git -C "$dir" apply --check "$patch" && git -C "$dir" apply "$patch" && echo "applied $2"; fi
}
mkdir -p external
clone_pinned DAAAM        https://github.com/MIT-SPARK/DAAAM.git     159f8daf1cf508b316c813890bdc164626842ff4
clone_pinned daaam-boxmot https://github.com/nicogorlo/boxmot.git    1dfa11e6a33c8172d07b3877a5f27a82ff8fb4cf
clone_pinned M2T2         https://github.com/NVlabs/M2T2.git         be2e5f6feae2f961615ab0f7b643f0f98b0c3a50
clone_pinned GraspGenX    https://github.com/NVlabs/GraspGenX.git    44eb1d77d6d27bc97ee361e3afb717a2115e30a9
clone_pinned molmospaces  https://github.com/allenai/molmospaces.git 3ad93ae1f77de995a87b5511bea18c655cfcb2ce
apply_patch external/molmospaces patches/eval_env/molmospaces.patch
if [ "${SKIP_ROBOCASA:-0}" != 1 ]; then
  # Source only (no `download_kitchen_assets`): saved scenes reference robosuite/robocasa
  # meshes, which scripts/pack_eval_episodes.py ships with the episodes.
  clone_pinned robosuite https://github.com/ARISE-Initiative/robosuite.git 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
  clone_pinned robocasa  https://github.com/robocasa/robocasa.git          4f8a2980def75a55dff96b990745b83540425f09
  apply_patch external/robosuite patches/eval_env/robosuite.patch
  apply_patch external/robocasa  patches/eval_env/robocasa.patch
fi

# ---------------------------------------------------------------- checkpoints (no venv needed)
fastsam=external/DAAAM/checkpoints/fastsam/FastSAM-s.pt
if [ ! -s "$fastsam" ]; then
  log "download FastSAM-s.pt"
  mkdir -p "$(dirname "$fastsam")"
  curl -L --fail -o "$fastsam.tmp" https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt
  mv "$fastsam.tmp" "$fastsam"
fi
echo "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844  $fastsam" | sha256sum -c -

# ---------------------------------------------------------------- .venv-sift
make_venv(){ # dir
  if [ -L "$1" ]; then die "$1 is a symlink into another tree; use a fresh clone"; fi
  [ -x "$1/bin/python" ] || uv venv -q --python "$PYBIN" "$1"
  grep -q "include-system-site-packages = false" "$1/pyvenv.cfg" || die "$1 must not use system site-packages"
}
log ".venv-sift: exact package set"
make_venv .venv-sift
uv pip install --python .venv-sift/bin/python --no-deps -r requirements-eval-sift.txt
uv pip install --python .venv-sift/bin/python --no-deps \
  "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9" \
  ./external/daaam-boxmot
uv pip install --python .venv-sift/bin/python --no-deps -e ./external/molmospaces -e .
uv pip check --python .venv-sift/bin/python || echo "(uv pip check reports the same unmet extras as the original env; see report)"

# ---------------------------------------------------------------- .venv-m2t2 (+ pointnet2_ops)
log ".venv-m2t2"
make_venv .venv-m2t2
uv pip install --python .venv-m2t2/bin/python -r requirements-eval-m2t2.txt
if ! .venv-m2t2/bin/python -c "import torch, pointnet2_ops._ext" 2>/dev/null; then
  nvcc_major(){ "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p'; }
  cuda_home=""
  if [ -n "${CUDA_HOME:-}" ] && [ "$(nvcc_major "$CUDA_HOME/bin/nvcc")" = 12 ]; then cuda_home=$CUDA_HOME
  elif command -v nvcc >/dev/null && [ "$(nvcc_major nvcc)" = 12 ]; then cuda_home=$(dirname "$(dirname "$(command -v nvcc)")")
  else
    cuda_home=$ROOT/.venv-eval-cache/cuda-12.6
    if [ ! -x "$cuda_home/bin/nvcc" ]; then
      log "no CUDA 12 nvcc found; fetching a minimal CUDA 12.6 compiler into $cuda_home"
      .venv-sift/bin/python scripts/fetch_cuda_toolkit.py "$cuda_home" --version 12.6
    fi
  fi
  log "build pointnet2_ops with $cuda_home ($("$cuda_home/bin/nvcc" --version | tail -1)), archs $TORCH_CUDA_ARCH_LIST"
  nv=$(.venv-m2t2/bin/python -c "import nvidia;print(list(nvidia.__path__)[0])")
  inc=$(find "$nv" -maxdepth 2 -type d -name include | paste -sd:)
  lib=$(find "$nv" -maxdepth 2 -type d -name lib | paste -sd:)
  (
    export CUDA_HOME=$cuda_home PATH="$cuda_home/bin:$ROOT/.venv-m2t2/bin:$PATH" MAX_JOBS=${MAX_JOBS:-8}
    export CPATH="$cuda_home/include:$inc${CPATH:+:$CPATH}" LIBRARY_PATH="$cuda_home/lib64:$lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    uv pip install --python .venv-m2t2/bin/python --no-build-isolation --no-deps --reinstall ./external/M2T2/pointnet2_ops
  )
fi

# ---------------------------------------------------------------- .venv-graspgenx
log ".venv-graspgenx"
make_venv .venv-graspgenx
uv pip install --python .venv-graspgenx/bin/python -r requirements-eval-graspgenx.txt
uv pip install --python .venv-graspgenx/bin/python --no-deps -e ./external/GraspGenX
# Checkpoints/grippers: GraspGenX would `git clone` (git-lfs) them on first import; fetch the
# same pinned revisions with huggingface_hub instead (a non-empty directory counts as installed).
.venv-graspgenx/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
ext = Path('external/GraspGenX/ext')
for repo, kind, rev, name in (('adithyamurali/GraspGenXModel', 'model', '7c834043c11a11417e31d6d5ea9355801e40a2c1', 'graspgenx_checkpoints'),
                              ('adithyamurali/gripper_descriptions', 'dataset', '19a03c00d19aeaf052d0f6801f0041982d676e8a', 'gripper_descriptions')):
    target = ext/name
    marker = target/'.snapshot_revision'
    if marker.is_file() and marker.read_text().strip() == rev:
        print('present', target); continue
    snapshot_download(repo_id=repo, repo_type=kind, revision=rev, local_dir=str(target))
    marker.write_text(rev + '\n'); print('downloaded', repo, rev, '->', target)
PY

# ---------------------------------------------------------------- HF model snapshots
log "HF snapshots into $HF_HOME"
.venv-sift/bin/python - "${SKIP_HF_MODELS:-0}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, constants
skip = sys.argv[1] == '1'
cache = Path(constants.HF_HUB_CACHE)
def pin_main(repo, rev):
    # The code loads models by name (revision "main"); point refs/main at the pinned
    # snapshot so offline runs (HF_HUB_OFFLINE=1) load exactly this revision.
    ref = cache/('models--' + repo.replace('/', '--'))/'refs'/'main'
    ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(rev)
print('m2t2 checkpoint:', hf_hub_download('wentao-yuan/m2t2', 'm2t2.pth', revision='a20325dcd19cd1b838274974d0ebe35bfe383796'))
pin_main('wentao-yuan/m2t2', 'a20325dcd19cd1b838274974d0ebe35bfe383796')
for repo, rev in (('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
                  ('allenai/Molmo2-4B', '042abfa7a38879a376cec03d949eff0aefaa0600')):
    if skip: print('SKIP_HF_MODELS=1: not downloading', repo); continue
    print(repo, '->', snapshot_download(repo, revision=rev)); pin_main(repo, rev)
PY

# ---------------------------------------------------------------- MolmoSpaces resources
# molmo_spaces' resource manager downloads robots (franka_droid), THOR objects, grasps, textures
# (fetchman) ... from its R2 bucket on first use and links them into MLSPACES_ASSETS_DIR. Without
# these variables it writes to ~/.cache (and the assets dir name hashes this checkout's path).
# Rebuilding a saved scene (load_saved_scene) needs them. Fetch once here (~13 GB, ~5 min) so
# offline compute nodes work.
export MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-$ROOT/.venv-eval-cache/molmo-spaces-resources}
export MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-$ROOT/.venv-eval-cache/molmospaces-assets}
if [ "${SKIP_MLSPACES_PREFETCH:-0}" != 1 ]; then
  log "MolmoSpaces resources into $MLSPACES_CACHE_DIR (links in $MLSPACES_ASSETS_DIR)"
  PYTHONPATH="$ROOT/external/molmospaces" .venv-sift/bin/python - <<'PY'
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR, get_resource_manager
import molmo_spaces.env.randomization.texture  # noqa: F401  (same trigger as scene loading)
get_resource_manager()
missing = [p for p in ('robots/franka_droid', 'textures/fetchman', 'objects/thor') if not (ASSETS_DIR/p).exists()]
assert not missing, f'MolmoSpaces resources missing under {ASSETS_DIR}: {missing}'
print('MolmoSpaces resources ready:', DATA_CACHE_DIR, '->', ASSETS_DIR)
PY
fi

# ---------------------------------------------------------------- eval_env.sh
cat > eval_env.sh <<EOF
# Generated by scripts/setup_eval_env.sh; source before running the eval or the services.
export EVAL_ROOT="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/external/DAAAM/src:$ROOT/external/molmospaces\${PYTHONPATH:+:\$PYTHONPATH}"
export HF_HOME="$HF_HOME"
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export ROBOCASA_DIR="$ROOT/external/robocasa/robocasa"
export MLSPACES_CACHE_DIR="$MLSPACES_CACHE_DIR"
export MLSPACES_ASSETS_DIR="$MLSPACES_ASSETS_DIR"
# Keep per-user tool state inside the checkout (ultralytics settings, CUDA/Triton JIT caches).
export YOLO_CONFIG_DIR="$ROOT/.venv-eval-cache/ultralytics" CUDA_CACHE_PATH="$ROOT/.venv-eval-cache/nv-compute" TRITON_CACHE_DIR="$ROOT/.venv-eval-cache/triton"
export M2T2_CHECKPOINT="$HF_HOME/hub/models--wentao-yuan--m2t2/snapshots/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth"
export FASTSAM_CHECKPOINT="$ROOT/external/DAAAM/checkpoints/fastsam/FastSAM-s.pt"
EOF
echo "wrote $ROOT/eval_env.sh"

# ---------------------------------------------------------------- self-check
if [ "${SKIP_CHECK:-0}" != 1 ]; then
  log "self-check"
  # shellcheck disable=SC1091
  source eval_env.sh
  .venv-sift/bin/python scripts/check_eval_env.py sift
  .venv-m2t2/bin/python scripts/check_eval_env.py m2t2
  .venv-graspgenx/bin/python scripts/check_eval_env.py graspgenx
fi
log "done"
