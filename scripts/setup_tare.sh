#!/usr/bin/env bash
# Isolated ROS 1 build; does not alter the simulator's Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."
revision=44500592b86138257273e0cab264e6a847ccefc7
mkdir -p .tools
# Avoid unpacking thousands of ROS/C++ headers on the AFS home filesystem.
export MAMBA_ROOT_PREFIX="$PWD/.tools/mamba"
if [[ ! -x .tools/bin/micromamba ]]; then
  curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest -o .tools/micromamba.tar.bz2
  tar -xjf .tools/micromamba.tar.bz2 -C .tools bin/micromamba
fi
if [[ ! -d external/tare_planner/.git ]]; then
  git clone --branch melodic-noetic https://github.com/caochao39/tare_planner.git external/tare_planner
  git -C external/tare_planner checkout "$revision"
fi
[[ $(git -C external/tare_planner rev-parse HEAD) == "$revision" ]] || { echo 'Unexpected TARE revision' >&2; exit 1; }
if [[ ! -x .venv-tare/bin/roscore ]]; then
  .tools/bin/micromamba create -y -p "$PWD/.venv-tare" --override-channels \
    -c robostack -c conda-forge --strict-channel-priority \
    ros-noetic-ros-base ros-noetic-pcl-ros ros-noetic-nav-msgs ros-noetic-tf \
    ros-noetic-visualization-msgs compilers cmake make pkg-config glog eigen 'python=3.9'
fi
mkdir -p .tools/tare_ws/src
ln -sfn "$PWD/external/tare_planner/src/tare_planner" .tools/tare_ws/src/tare_planner
.tools/bin/micromamba run -p "$PWD/.venv-tare" bash -c '
  cd .tools/tare_ws
  catkin_make -j4 -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOpenGL_GL_PREFERENCE=LEGACY -DOPENGL_gl_LIBRARY=/usr/lib/x86_64-linux-gnu/libGL.so \
    -DOPENGL_INCLUDE_DIR=/usr/include -DCMAKE_CXX_FLAGS="-include stack" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/libGL.so"
'
