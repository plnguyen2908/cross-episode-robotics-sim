#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export TORCH_HOME="$ROOT/.venv-vggt-cache"
export HF_HOME="$ROOT/.venv-vggt-cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.venv-vggt-cache/xdg"
export MPLCONFIGDIR="$ROOT/.venv-vggt-cache/matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
STREAM=${STREAM:-runs/slam_stream_masked/sensors}
OUTPUT=${OUTPUT:-$(mktemp -d "$ROOT/runs/vggt_comparison_XXXXXX")}
echo "OUTPUT=$OUTPUT"
for CASE in complete dropout dense dense_masked dense_dropout dense_masked_dropout; do
  EXTRA=()
  if [[ $CASE == *dropout ]]; then EXTRA+=(--dropout 300 310); fi
  if [[ $CASE == dense* ]]; then EXTRA+=(--min-disparity 10 --max-keyframe-gap 10); fi
  if [[ $CASE == *masked* ]]; then EXTRA+=(--mask-robot); fi
  .venv-vggt/bin/python -u scripts/compare_vggt_stream.py \
    --upstream external/vggt-slam --stream "$STREAM" \
    --weights "$TORCH_HOME/hub/checkpoints/model.pt" \
    --output "$OUTPUT/$CASE" "${EXTRA[@]}" 2>&1 | tee "$OUTPUT/$CASE.log"
  .venv-vggt/bin/python scripts/score_vggt_stream.py \
    --run "$OUTPUT/$CASE" --stream "$STREAM" \
    --diagnostics runs/slam_stream_masked/diagnostics \
    --baseline runs/slam_hybrid_planar
done
