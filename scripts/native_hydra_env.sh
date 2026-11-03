#!/usr/bin/env bash
# Source from repository root; separate env keeps native dependencies isolated.
export PATH="$PWD/.venv-hydra/bin:$PATH"
export CC="$PWD/.venv-hydra/bin/x86_64-conda-linux-gnu-cc"
export CXX="$PWD/.venv-hydra/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$PWD/.tools/hydra_ws/install:$PWD/.venv-hydra"
export PKG_CONFIG_PATH="$PWD/.tools/hydra_ws/install/lib/pkgconfig:$PWD/.venv-hydra/lib/pkgconfig"
export LD_LIBRARY_PATH="$PWD/.tools/hydra_ws/install/lib:$PWD/.venv-hydra/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${HYDRA_BUILD_JOBS:-4}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
