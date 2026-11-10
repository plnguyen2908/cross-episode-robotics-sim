#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=53e9966d7a8e2fa7494676772bccc35280f5c0ed
if [[ ! -d external/cap-x/.git ]]; then
  git clone https://github.com/capgym/cap-x.git external/cap-x
fi
if [[ "$(git -C external/cap-x rev-parse HEAD)" != "$revision" ]]; then
  if [[ -n "$(git -C external/cap-x status --porcelain)" ]]; then
    echo 'Existing CaP-X checkout has changes; refusing to overwrite.' >&2
    exit 1
  fi
  git -C external/cap-x checkout "$revision"
fi
patch_path="$PWD/patches/capx_external_only.patch"
if git -C external/cap-x apply --reverse --check "$patch_path" 2>/dev/null; then
  echo 'CaP-X external-environment import patch already applied.'
else
  git -C external/cap-x apply --check "$patch_path"
  git -C external/cap-x apply "$patch_path"
fi
