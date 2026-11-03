#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Component integration only: do not install upstream's full CUDA/ROS/Hydra stack
# over the working simulation environment.
mkdir -p external
if [ ! -d external/DAAAM/.git ]; then
  git clone https://github.com/MIT-SPARK/DAAAM.git external/DAAAM
  git -C external/DAAAM checkout 159f8daf1cf508b316c813890bdc164626842ff4
fi
if [ ! -d external/daaam-boxmot/.git ]; then
  git clone https://github.com/nicogorlo/boxmot.git external/daaam-boxmot
  git -C external/daaam-boxmot checkout 1dfa11e6a33c8172d07b3877a5f27a82ff8fb4cf
fi
test "$(git -C external/DAAAM rev-parse HEAD)" = 159f8daf1cf508b316c813890bdc164626842ff4 || { echo 'Unexpected DAAAM revision; inspect before changing it.' >&2; exit 1; }
test "$(git -C external/daaam-boxmot rev-parse HEAD)" = 1dfa11e6a33c8172d07b3877a5f27a82ff8fb4cf || { echo 'Unexpected boxmot revision; inspect before changing it.' >&2; exit 1; }
.venv-slam/bin/python -m pip install --no-deps 'ultralytics==8.3.221' ./external/daaam-boxmot
.venv-slam/bin/python -m pip install -r requirements-perception.txt filterpy gdown lapx loguru yacs ftfy ultralytics-thop num2words python-dotenv
checkpoint=external/DAAAM/checkpoints/fastsam/FastSAM-s.pt
mkdir -p "$(dirname "$checkpoint")"
if [ ! -s "$checkpoint" ]; then
  curl -L --fail -o "$checkpoint.tmp" https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt
  mv "$checkpoint.tmp" "$checkpoint"
fi
