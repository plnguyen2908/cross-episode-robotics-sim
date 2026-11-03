#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/nobackup2/le/.cache/uv-openpi}"
# Keep upstream inference's dependency set intact; the training wrapper opts into
# the official v3 reader and its datasets dependency from this separate directory.
uv pip install --python .venv-openpi/bin/python --no-deps \
  --target external/lerobot-v3 lerobot==0.4.3 datasets==4.1.1
uv pip install --python .venv-openpi/bin/python --no-deps accelerate==1.10.1
