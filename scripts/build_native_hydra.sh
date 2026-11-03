#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-hydra/bin:$PATH"
export CC="$PWD/.venv-hydra/bin/x86_64-conda-linux-gnu-cc"
export CXX="$PWD/.venv-hydra/bin/x86_64-conda-linux-gnu-c++"
export PKG_CONFIG_PATH="$PWD/.tools/hydra_ws/install/lib/pkgconfig:$PWD/.venv-hydra/lib/pkgconfig"
export LD_LIBRARY_PATH="$PWD/.tools/hydra_ws/install/lib:$PWD/.venv-hydra/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
prefix="$PWD/.tools/hydra_ws/install"
base="$PWD/.venv-hydra"
build() {
 local name="$1" source="$2"
 shift 2
 cmake -S "$source" -B ".tools/hydra_ws/build_py311/$name" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$prefix;$base" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_RPATH="$prefix/lib;$base/lib" "$@"
 cmake --build ".tools/hydra_ws/build_py311/$name" -j"${HYDRA_BUILD_JOBS:-4}"
 cmake --install ".tools/hydra_ws/build_py311/$name"
}
build config_utilities .tools/hydra_ws/src/config_utilities/config_utilities -DCONFIG_UTILS_BUILD_DEMOS=OFF -DCONFIG_UTILS_ENABLE_ROS=OFF
build spatial_hash .tools/hydra_ws/src/spatial_hash
build spark_dsg .tools/hydra_ws/src/spark_dsg -DSPARK_DSG_BUILD_EXAMPLES=OFF -DSPARK_DSG_BUILD_PYTHON=OFF -DSPARK_DSG_BUILD_ZMQ=OFF
build kimera_rpgo .tools/hydra_ws/src/kimera_rpgo -DKIMERA_RPGO_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
build pose_graph_tools .tools/hydra_ws/src/pose_graph_tools/pose_graph_tools
build kimera_pgmo .tools/hydra_ws/src/kimera_pgmo/kimera_pgmo
build teaserpp .tools/hydra_ws/src/teaser_plusplus -DBUILD_DOC=OFF -DBUILD_TEASER_FPFH=OFF
build hydra external/Hydra -DHYDRA_ENABLE_EVAL=OFF -DHYDRA_ENABLE_PYTHON=ON -DPython_EXECUTABLE="$base/bin/python"
