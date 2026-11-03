#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=7b658d10b2273ea8add988163290e390d3711f13
if [[ ! -d external/OneRING/.git ]]; then
  git clone https://github.com/Ainaz99/OneRING.git external/OneRING
  git -C external/OneRING checkout "$revision"
fi
[[ "$(git -C external/OneRING rev-parse HEAD)" == "$revision" ]] || { echo 'OneRING revision differs; preserve existing checkout and resolve explicitly.' >&2; exit 1; }
patch_path="$PWD/scripts/onering_patches/runtime_compatibility.patch"
if git -C external/OneRING apply --check "$patch_path"; then
  git -C external/OneRING apply "$patch_path"
else
  git -C external/OneRING apply --reverse --check "$patch_path"
fi
.venv-slam/bin/python -m venv --system-site-packages .venv-ring
.venv-slam/bin/python - <<'PY'
import site,sys
from pathlib import Path
root=Path('.venv-ring/lib')/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
paths=[str(Path(p).resolve()) for p in site.getsitepackages()]
(root/'sensor_runtime.pth').write_text('\n'.join(paths)+'\n')
PY
.venv-ring/bin/python -m pip install --no-deps -r requirements-onering.txt
.venv-ring/bin/python -m pip install --no-deps --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+d3370a1dabb80fde5793c92c9e178c80aa91d801
.venv-slam/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('AinazEftekhar/OneRING','ring_model_step_40356421.ckpt',revision='adf715b75c11ca789a033c9f157360f50c48a6e0'))
PY
