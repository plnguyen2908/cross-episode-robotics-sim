#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/nobackup2/le/.cache/uv-openpi}"
if [[ ! -d external/M2T2 ]]; then git clone --depth 1 https://github.com/NVlabs/M2T2.git external/M2T2; fi
# Reuse the local torch 2.7.1/cu126 installation read-only; install additions separately.
if [[ ! -d .venv-m2t2 ]]; then uv venv --python .venv-openpi/bin/python .venv-m2t2; fi
printf '%s\n' "$PWD/.venv-openpi/lib/python3.11/site-packages" > .venv-m2t2/lib/python3.11/site-packages/openpi_dependencies.pth
uv pip install --python .venv-m2t2/bin/python hydra-core h5py trimesh scipy ninja setuptools zstandard
export CUDA_HOME=/nobackup2/le/.cache/cuda-m2t2-12.6
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then .venv-m2t2/bin/python scripts/setup_m2t2_cuda.py; fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" MAX_JOBS=4
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$(find "$PWD/.venv-openpi/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name include | paste -sd:)"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
uv pip install --python .venv-m2t2/bin/python --no-build-isolation --no-deps external/M2T2/pointnet2_ops
.venv-m2t2/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('wentao-yuan/m2t2','m2t2.pth',revision='a20325dcd19cd1b838274974d0ebe35bfe383796',cache_dir='/nobackup2/le/.cache/hub'))
PY
