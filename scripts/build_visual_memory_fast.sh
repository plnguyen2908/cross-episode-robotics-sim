#!/usr/bin/env bash
# Map-only balanced history preprocessing; no Gemini or local VLM services.
set -euo pipefail
CODE_ROOT=${MEMORY_CODE_ROOT:-/nobackup/le/cross-episode-qwen-capx}
PY=${MEMORY_PYTHON:-$CODE_ROOT/.venv-sift/bin/python}
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/external/Hierarchical-Localization${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_ROOT"
exec "$PY" -m cross_episode_sim.mapping.fast_visual_memory \
  --output "${VISUAL_MEMORY_OUTPUT:-$CODE_ROOT/runs/visual_memory_balanced_v4}" "$@"
