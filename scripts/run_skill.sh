#!/usr/bin/env bash
# Run one atomic-skill demonstration on its validated starting scene.
#
#   scripts/run_skill.sh <skill> [output_dir] [extra controller flags...]
#
# Skills: cross-room, cabinet-door, cabinet-transfer, drawer, drawer-loop,
#         drawer-pick-place, stove-knob, faucet, microwave-button, toaster-lever,
#         toaster-insertion, oven-rack, oven-pick-place, blender-lid
set -euo pipefail

skill="${1:?usage: scripts/run_skill.sh <skill> [output_dir] [flags...]}"
shift
output="${1:-runs/${skill}_$(date +%Y%m%d_%H%M%S)}"
[[ $# -gt 0 ]] && shift

data="${CES_DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
honey=(--recording "$data/skill_scenes/saved_grasp_reuse/robocasa__lightwheel__honey_bottle__HoneyBottle003__2001/initial_scene"
       --asset lightwheel/honey_bottle/HoneyBottle003 --grasp-source qualified)
egg=(--recording "$data/skill_scenes/saved_grasp_reuse/molmo__Egg_14__2002/initial_scene"
     --asset Egg_14 --grasp-source qualified)
bread=(--recording "$data/skill_scenes/toaster_bread_initial"
       --asset lightwheel/sandwich_bread/SandwichBread005 --grasp-source surface)

case "$skill" in
  cross-room|cabinet-door|drawer|stove-knob|faucet|microwave-button|oven-rack) scene=("${honey[@]}") ;;
  cabinet-transfer|drawer-loop|drawer-pick-place|oven-pick-place) scene=("${egg[@]}") ;;
  blender-lid) scene=("${egg[@]/qualified/molmo}") ;;
  toaster-lever|toaster-insertion) scene=("${bread[@]}") ;;
  *) echo "unknown skill: $skill" >&2; exit 2 ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$(dirname "$output")"
python -u -m cross_episode_sim.manipulation.robocasa "${scene[@]}" \
  --approach-policy any-above-table "--$skill" --output "$output" "$@"
