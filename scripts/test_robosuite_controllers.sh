#!/usr/bin/env bash
set -euo pipefail
echo "Replaced by MolmoSpaces controllers (qualified absolute-position replay only)."
echo "Legacy robosuite conversion experiments remain available via scripts/test_controller_tasks.py --backend robosuite."
exec bash "$(dirname "$0")/test_molmospaces_controllers.sh" "$@"
