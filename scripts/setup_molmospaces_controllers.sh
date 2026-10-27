#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=3ad93ae1f77de995a87b5511bea18c655cfcb2ce
repo=external/molmospaces
if [[ ! -d "$repo/.git" ]]; then
  git clone https://github.com/allenai/molmospaces.git "$repo"
  git -C "$repo" checkout --detach "$revision"
fi
[[ $(git -C "$repo" rev-parse HEAD) == "$revision" ]] || {
  echo "MolmoSpaces revision mismatch; refusing to overwrite an existing checkout." >&2; exit 1;
}
patchfile="$PWD/patches/molmospaces_unlimited_control_range.patch"
if git -C "$repo" apply --reverse --check "$patchfile" 2>/dev/null; then
  echo "MolmoSpaces controller fix already applied."
else
  git -C "$repo" apply --check "$patchfile"
  git -C "$repo" apply "$patchfile"
fi
