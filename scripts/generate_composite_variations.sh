#!/usr/bin/env bash
# 35 seeded mixed-variation demos; eight independent episode workers with GPU leases.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_dir/scripts/generate_composite_pilot.sh" \
  --suite randomized --workers 8 "$@"
