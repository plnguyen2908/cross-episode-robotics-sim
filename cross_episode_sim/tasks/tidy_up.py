"""Tidy up: return a stray mug to the kitchen and a book to the living room.

A small composite task kept as a template for new tasks (docs/adding_a_task.md).
It reuses the breakfast house and its qualified objects, and composes two
cross-room transfers from existing transactional skills:

1. prepare: choose task objects from a prepared scene; everything else stays
   in the scene as background;
2. controller: the breakfast episode controller already provides the
   `transfer_role` skill (navigate, grasp, carry, place, withdraw) and the
   measured check `verify_role`;
3. composite: a list of operations with arguments, run by CompositeEpisode;
4. goal: measured predicates that must all hold at the end.

    python -m cross_episode_sim.tasks.tidy_up --output runs/tidy_up
"""

import argparse
import json
from pathlib import Path

from cross_episode_sim.paths import DATA_DIR
from cross_episode_sim.skills.composite import CompositeEpisode, Operation
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode, build_controller

BASE_EPISODE = DATA_DIR / "base_episodes/breakfast_three_room"
# Task roles in the base scene, in the order the robot handles them.
ROLES = ("misplaced_cup", "reading")
INSTRUCTION = (
    "Tidy up. Take the mug from the living-room side table to the kitchen counter, "
    "then take the book from the dining table to the living-room side table. "
    "Leave everything else where it is and finish with an empty gripper."
)


def prepare(output, base=BASE_EPISODE):
    """Write a task manifest that keeps ROLES as task objects and the rest as background."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(base) / "task_manifest.json").read_text())
    task_objects = [b for b in manifest["bindings"] if b["role"] in ROLES]
    others = [dict(body=b["body"], room=b["source"], kind="pickable", key=b["key"], task_object=False,
                   position=b["position"], initial_quaternion=b["initial_quaternion"])
              for b in manifest["bindings"] if b["role"] not in ROLES]
    task_objects.sort(key=lambda b: ROLES.index(b["role"]))
    manifest.update(task="tidy_up", bindings=task_objects, background=manifest["background"] + others,
                    instruction=INSTRUCTION)
    (output / "task_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "instruction.txt").write_text(INSTRUCTION + "\n")
    return manifest


class TidyUp(BreakfastEpisode):
    def goal(self):
        evidence = {role: self.verify_role(role) for role in ROLES}
        evidence["empty_hand"] = not self.attached and not getattr(self, "holding_loaf", False)
        return evidence

    def run_task(self):
        operations = {
            "check_scene": Operation(self.validate_population, lambda: True),
            "transfer": Operation(self.transfer_role, self.verify_role),
        }
        steps = [dict(operation="check_scene", arguments={})]
        steps += [dict(operation="transfer", arguments=dict(role=role)) for role in ROLES]
        return CompositeEpisode(self, operations, self.goal).run("tidy_up", steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = prepare(output)
    controller = build_controller(output, manifest, episode_class=TidyUp)
    return 0 if controller.run_task() else 1


if __name__ == "__main__":
    raise SystemExit(main())
