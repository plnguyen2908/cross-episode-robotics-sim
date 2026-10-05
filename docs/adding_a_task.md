# Adding a task

A task is four things:

1. **a prepared scene**: which objects are task objects, where they start and where they go;
2. **a controller**: a class that loads the scene with the robot and provides skills;
3. **a composite plan**: a list of skill calls with arguments;
4. **a goal**: measured predicates that must all be true at the end.

`cross_episode_sim/tasks/tidy_up.py` implements a complete small task in about 80 lines, and
the rest of this page walks through it. Run it with

```bash
python -m cross_episode_sim.tasks.tidy_up --output runs/tidy_up
```

## 1. Prepare the scene

Scenes are prepared once, written to the output directory as `task_manifest.json`, and never
changed by the controller. A manifest names the scene file and lists **bindings** (task objects)
and **background** objects:

```json
{
  "scene_xml": ".../robocasa_scene.xml",
  "bindings": [
    {"role": "misplaced_cup", "body": "misplaced_cup_test_object_main", "asset": "Mug_1",
     "source": "living", "destination": "kitchen", "destination_position": [x, y, z],
     "grasp_path": ".../misplaced_cup_grasps.npz", "initial_quaternion": [w, x, y, z], ...}
  ],
  "background": [{"body": "...", "room": "kitchen", "kind": "pickable", ...}],
  "supports": {"kitchen": "counter_main_main_group_main", "dining": "dining_table_dining_room_main", ...},
  "instruction": "..."
}
```

`tidy_up.prepare()` starts from the prepared breakfast house in the data bundle, keeps two
objects as bindings and turns the others into background, so they stay in the scene as
obstacles.

For a new scene, follow `tasks/breakfast/scene.py`:

- `bind_roles()` picks objects for each role from the qualified grasp registry
  (`data/grasp_sweep/current_results.json`), so every task object has grasps that were physically
  verified on this robot;
- `allocate_sites()` reserves reachable, separated start and goal spots on support surfaces,
  with clearance for the robot to dock, grasp and withdraw;
- `settle_population()` steps physics until objects rest and rejects scenes where anything
  moved, interpenetrates or lost its support;
- the result is written as a scene XML plus manifest.

Keep authoring randomness behind explicit seeds. A task is only meaningful if the same seed
gives the same scene.

## 2. The controller

Subclass an existing controller rather than starting from scratch. `BreakfastEpisode`
(`tasks/breakfast/episode.py`) provides, for any manifest:

| Method | What it does |
|---|---|
| `transfer_role(role)` | navigate to the object, select and qualify a grasp, lift, tuck, carry to the destination room, place upright, release, withdraw, tuck |
| `verify_role(role)` | measured: object supported by the destination surface, within 10 cm of its target, upright within 20°, not held |
| `validate_population()` | checks initial supports and that no objects interpenetrate |
| `navigate_to_site(room, point, carrying)` | docks in front of a point, trying alternative docks |

Fixture skills come from `fixtures/` (doors, drawers, knobs, buttons, …) and `skills/fixtures.py`.
`build_controller(output, manifest, episode_class)` loads the scene, adds the robot at its spawn
pose and returns the controller.

## 3. The composite plan

`CompositeEpisode` runs named operations in order. Each `Operation` pairs an action with its
measured postcondition:

```python
operations = {
    "check_scene": Operation(self.validate_population, lambda: True),
    "transfer": Operation(self.transfer_role, self.verify_role),
}
steps = [dict(operation="check_scene", arguments={}),
         dict(operation="transfer", arguments=dict(role="misplaced_cup")),
         dict(operation="transfer", arguments=dict(role="reading"))]
CompositeEpisode(self, operations, self.goal).run("tidy_up", steps)
```

All step arguments are validated against the operation signatures before physics starts. The
plan is plain data (`composite_execution.json` records it with timing and outcome), so a planner
or a memory module can produce it.

## 4. The goal

```python
def goal(self):
    evidence = {role: self.verify_role(role) for role in ROLES}
    evidence["empty_hand"] = not self.attached and not getattr(self, "holding_loaf", False)
    return evidence
```

The episode succeeds only if every value is `True`. Goals must be **measured** from simulator
state (contacts, poses, joint positions, particle positions), never taken from what the
controller believes it did.

## Writing a new skill

When a task needs an interaction that no existing skill covers, implement it as an
`AtomicSkill` so it gets checkpointing and rollback for free:

```python
from cross_episode_sim.skills.atomic import AtomicSkill

# Sketch: the controller helpers called here stand for whatever your fixture needs;
# fixtures/microwave_button.py is a complete button implementation.
class PressButton(AtomicSkill):
    def __init__(self, controller, button):
        super().__init__(controller, f"press {button}")
        self.button = button

    def candidates(self):
        # Bounded alternatives to try, e.g. docks around the appliance.
        return list(self.controller.dock_candidates_for(self.button))

    def execute_candidate(self, dock):
        c = self.controller
        c.task_navigate(dock[:2], carrying=False, face=dock[2])
        c.press_with_fingertip(self.button)        # physical contact only
        c.withdraw_and_tuck()

    def verify(self):
        return self.controller.button_state(self.button) == "pressed"
```

`skill.run()` tries candidates in order. Before each try it checkpoints the simulator; if
`execute_candidate` raises or `verify` returns false, it saves diagnostics under
`rejected_trials/`, restores the checkpoint and moves on. Add `snapshot_skill_state` /
`restore_skill_state` to the controller for any extra Python state a rollback must restore.
Wrap an existing method without writing a class using
`CallbackSkill(controller, name, candidates, execute, verify)`.

Rules that keep demonstrations honest:

- Move objects and fixtures only through robot contact on their native joints. Never set object
  poses or weld objects to the gripper during execution.
- Fixture state (door angle, button travel, knob angle) is read from joints, not assumed.
- Put any non-robot event (a person filling a cup, a machine finishing) in its own operation and
  record it in `report.json`.

## Exposing the task to policies

Register the task in `cross_episode_sim/env/tasks.py` (tidy_up is registered there):

```python
def build_tidy_up(output, seed, options):
    from cross_episode_sim.tasks.breakfast.episode import build_controller
    from cross_episode_sim.tasks.tidy_up import TidyUp, prepare
    return build_controller(output, prepare(output), episode_class=TidyUp)

def tidy_up_goals(controller):
    goals = {role: controller.verify_role(role) for role in ROLES}
    bodies = {controller.model.body(i["body"]).id for i in controller.manifest["bindings"]}
    goals["empty_hand"] = not finger_contact_bodies(controller) & bodies
    return goals

TASKS["tidy_up"] = TaskSpec("tidy_up", build_tidy_up, tidy_up_goals)
```

Policy goals must not read the demonstrator's bookkeeping, such as `attached` or
`holding_loaf`, because a learned policy never sets them. Replace such terms with measurements,
for example `env.tasks.finger_contact_bodies()` for "the hand is empty". Supply `after_step`
if the task has events that are not robot actions.

## Checklist

- [ ] `--prepare-only` produces the same manifest for the same seed.
- [ ] The demonstration succeeds on several seeds; `report.json` shows the goal evidence.
- [ ] `rejected_trials/` is small. Many rejections point to tight docks or placements.
- [ ] The video shows no teleports or objects moving without contact.
- [ ] Unit tests cover new geometry or state logic (see `tests/`).
