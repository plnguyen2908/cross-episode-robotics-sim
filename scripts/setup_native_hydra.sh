#!/usr/bin/env bash
# Isolated upstream C++/Python mapping backend. Does not alter simulator env.
set -euo pipefail
cd "$(dirname "$0")/.."
MAMBA="${MAMBA_EXE:-$PWD/.tools/bin/micromamba}"
if [[ ! -x "$MAMBA" ]]; then
  echo 'Set MAMBA_EXE to a micromamba executable.' >&2; exit 1
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PWD/.tools/mamba}"
if [[ ! -x .venv-hydra/bin/python ]]; then
  "$MAMBA" create -y -p "$PWD/.venv-hydra" --file configs/navigation/hydra_environment_linux64.lock
fi
.venv-hydra/bin/python - <<'PY'
import json,subprocess
from pathlib import Path
for name,repo in json.loads(Path('configs/navigation/hydra_sources.json').read_text()).items():
    path=Path(repo['path'])
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init',str(path)],check=True)
        subprocess.run(['git','-C',str(path),'remote','add','origin',repo['url']],check=True)
        subprocess.run(['git','-C',str(path),'fetch','--depth','1','origin',repo['revision']],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach','FETCH_HEAD'],check=True)
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=repo['revision']:raise RuntimeError(f'{path}: unexpected revision {actual}')
    patch=Path('scripts/hydra_patches')/f'{name}.patch'
    if patch.exists() and patch.stat().st_size:
        applied=subprocess.run(['git','-C',str(path),'apply','--reverse','--check',str(patch.resolve())],capture_output=True)
        if applied.returncode:subprocess.run(['git','-C',str(path),'apply',str(patch.resolve())],check=True)
prefix=Path('.venv-hydra').resolve()
# Conda glog omits the pkg-config file required by upstream Spatial-Hash.
version=json.loads(next((prefix/'conda-meta').glob('glog-*.json')).read_text())['version']
p=Path('.tools/hydra_ws/install/lib/pkgconfig/libglog.pc');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(f'prefix={prefix}\nlibdir=${{prefix}}/lib\nincludedir=${{prefix}}/include\nName: libglog\nDescription: Conda glog compatibility metadata\nVersion: {version}\nLibs: -L${{libdir}} -lglog\nCflags: -I${{includedir}} -DGLOG_USE_GLOG_EXPORT\n')
PY
bash scripts/build_native_hydra.sh
source scripts/native_hydra_env.sh
python -m pip install click distinctipy imageio networkx pyyaml scipy tqdm matplotlib shapely pyzmq \
  'spark_dataset_interfaces @ git+https://github.com/MIT-SPARK/spark_dataset_interfaces'
python -m pip install --no-deps .tools/hydra_ws/src/spark_dsg
python -m pip install --no-deps external/Hydra
python -c 'import spark_dsg, hydra_python; print("Native Hydra bindings imported")'
