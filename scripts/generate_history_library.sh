#!/usr/bin/env bash
# Independent GPU workers; each targets COUNT successful histories.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
COUNT="${COUNT:-30}"
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [batch-directory-to-create-or-resume]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    batch_root="$1"
    mkdir -p "$batch_root"
else
    mkdir -p runs
    batch_root="$(mktemp -d runs/history_batches_XXXXXX)"
fi
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'cleanup; exit 130' INT TERM
launch() {
    local task="$1" seed="$2" position="$3"
    local mode=--output
    [[ -f "$batch_root/$task/batch.json" ]] && mode=--resume
    "$PYTHON_BIN" -u -m "scripts.generate_${task}_histories" \
        "$mode" "$batch_root/$task" --count "$COUNT" --start-seed "$seed" --gpu auto --progress-position "$position" \
        >> "$batch_root/$task.log" &
    pids+=("$!")
    echo "$task with automatic GPU assignment: $batch_root/$task.log"
}
launch coffee "${COFFEE_START_SEED:-100}" 0
launch breakfast "${BREAKFAST_START_SEED:-1000}" 1
echo "Resume with: bash scripts/generate_history_library.sh '$batch_root'"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
