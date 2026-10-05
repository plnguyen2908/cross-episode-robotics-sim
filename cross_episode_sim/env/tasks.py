"""Task hooks used by `TaskEnv`: how to build an episode, check goals and fire events.

A task is registered as a `TaskSpec`:

- `build(output, seed, options)` prepares a fresh episode directory and returns
  the task controller with the scene loaded and the robot at its start pose.
- `goals(controller)` returns measured goal predicates; the episode succeeds
  when all are true.
- `after_step(controller)` runs after every control step and fires
  non-robot events, such as a person filling the gathered vessels.

Goals must be measurable from simulator state. They must not read the
scripted demonstrator's own bookkeeping (which object it believes it holds,
which phase it is in), because a learned policy does not update it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    build: Callable[[Any, int | None, dict], Any]
    goals: Callable[[Any], dict]
    after_step: Callable[[Any], None] = lambda controller: None


def finger_contact_bodies(controller):
    """Bodies (other than the robot) touched by a gripper pad."""
    model, data = controller.model, controller.data
    pads = {model.body(controller.profile.namespace + name).id for name in controller.profile.finger_bodies}
    touched = set()
    for contact in data.contact[: data.ncon]:
        a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if a in pads and b not in pads:
            touched.add(b)
        elif b in pads and a not in pads:
            touched.add(a)
    return touched


def room_of(controller, info):
    """Task room whose support currently holds the object, or None (held, fallen)."""
    try:
        return controller.current_room(info)
    except StopIteration:
        return None


# -- breakfast: gather empty vessels, a person fills them, serve them ------
def build_breakfast(output, seed, options):
    from cross_episode_sim.tasks.breakfast.episode import build_controller
    from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast, prepare_episode

    manifest = prepare_episode(output, seed=seed, **options)
    controller = build_controller(output, manifest, episode_class=GatherBreakfast)
    controller.configure_phase("GATHER")
    controller.policy_driven = True
    return controller


def breakfast_after_step(controller):
    """Fill the vessels once all of them stand on kitchen filling surfaces.

    The demonstrator also steps the robot back before the person fills; under a
    policy the robot simply has to have released every vessel.
    """
    if controller.phase != "GATHER" or controller.filled():
        return
    vessels = {controller.model.body(i["body"]).id for i in controller.manifest["bindings"]}
    in_hand = finger_contact_bodies(controller) & vessels
    if in_hand or not all(controller.is_filling_site(room_of(controller, i)) for i in controller.manifest["bindings"]):
        return
    controller.fill_physical_contents()
    controller.configure_phase("SERVE")


def breakfast_goals(controller):
    goals = {i["role"]: controller.phase == "SERVE" and controller.verify_role(i["role"])
             for i in controller.manifest["bindings"]}
    from cross_episode_sim.tasks.breakfast.storage import storage_closed

    goals["filled"] = controller.filled()
    goals["storage_closed"] = storage_closed(controller)
    goals["empty_hand"] = not finger_contact_bodies(controller) & {
        controller.model.body(i["body"]).id for i in controller.manifest["bindings"]}
    return goals


# -- tidy up: the template task from docs/adding_a_task.md -----------------
def build_tidy_up(output, seed, options):
    from cross_episode_sim.tasks.breakfast.episode import build_controller
    from cross_episode_sim.tasks.tidy_up import TidyUp, prepare

    return build_controller(output, prepare(output), episode_class=TidyUp)


def tidy_up_goals(controller):
    from cross_episode_sim.tasks.tidy_up import ROLES

    goals = {role: controller.verify_role(role) for role in ROLES}
    bodies = {controller.model.body(i["body"]).id for i in controller.manifest["bindings"]}
    goals["empty_hand"] = not finger_contact_bodies(controller) & bodies
    return goals


TASKS = {
    "breakfast": TaskSpec("breakfast", build_breakfast, breakfast_goals, breakfast_after_step),
    "tidy_up": TaskSpec("tidy_up", build_tidy_up, tidy_up_goals),
}
