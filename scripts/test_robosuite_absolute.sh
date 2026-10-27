#!/usr/bin/env bash
# Compatibility entry point: native MolmoSpaces replaces the robosuite replay.
set -euo pipefail
echo "This entry point now uses MolmoSpaces controllers."
exec bash "$(dirname "$0")/test_molmospaces_absolute.sh" "$@"
