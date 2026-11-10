#!/usr/bin/env bash
# Extend the existing sensor environment without upgrading its dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv-slam/bin/python -m venv .venv-sift
.venv-slam/bin/python - <<'PY'
import sys
from pathlib import Path
root=Path('.venv-sift/lib')/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
paths=list(dict.fromkeys(str(Path(p).resolve()) for p in sys.path if p and 'site-packages' in p))
(root/'robot_dependencies.pth').write_text('\n'.join(paths)+'\n')
PY
.venv-sift/bin/python -m pip install --no-deps pycolmap-cuda12==4.2.1
.venv-sift/bin/python - <<'PY'
import pycolmap, open3d
assert pycolmap.has_cuda, 'CUDA COLMAP is required'
assert open3d.core.cuda.is_available(), 'CUDA Open3D is required'
print('CUDA SLAM dependencies ready')
PY
