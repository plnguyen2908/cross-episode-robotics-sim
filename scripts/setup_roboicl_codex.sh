#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$root/external/roboicl"
patch="$root/patches/roboicl_codex_cli.patch"
if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
  echo 'Codex transport patch already applied.'
else
  git -C "$repo" apply --check "$patch"
  git -C "$repo" apply "$patch"
fi
cd "$repo"
python3 -m unittest discover -s tests -p test_codex_transport_offline.py
