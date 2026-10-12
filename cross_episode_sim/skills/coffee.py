"""Coffee recipes assembled from reusable physical skills and explicit events."""

import numpy as np

from cross_episode_sim.skills.atomic import AtomicSkill, BoundSkill
from cross_episode_sim.skills.portafilter import body_pose
from cross_episode_sim.skills.sequence import SkillSequence
from cross_episode_sim.tasks.coffee.machine import CENTER, PORTAFILTER, RIM


class PickDosingBoxSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "pick_dosing_box")

    def candidates(self):
        return (None,)

    def execute_candidate(self, candidate):
        c = self.controller
        c.review_phase = "LOAD PORTAFILTER FROM SMALL BOX"
        c.pickup(c.spec["dosing_body"], np.asarray(c.manifest["box_grasp"]))
        lift = c.bread_pose().copy()
        lift[2, 3] += 0.10
        c.held_pose("lift grounds box", lift)
        c.allow_support = False

    def verify(self):
        c = self.controller
        return bool(
            c.object_name == c.spec["dosing_body"]
            and c.holding_loaf
            and len(c.contacts()) >= 2
            and not c.inventory()["spilled"]
        )


class PourGroundsSkill(AtomicSkill):
    def __init__(self, controller, *, prepare=False, track_portafilter=False):
        super().__init__(controller, "pour_grounds")
        self.prepare = prepare
        self.track_portafilter = track_portafilter

    def candidates(self):
        return (None,)

    def execute_candidate(self, candidate):
        c = self.controller
        if not c.holding_loaf or len(c.contacts()) < 2:
            raise RuntimeError("Pour requires a physically held dosing vessel")
        if self.prepare:
            c.prepare_loaded_manipulation()
        if self.track_portafilter:
            source = np.asarray(c.spec["portafilter_original_pose"])
            center = body_pose(c.data, PORTAFILTER) @ np.linalg.inv(source) @ np.array([*CENTER, RIM, 1.0])
            c.spec["hopper"].update(center_xy=center[:2].tolist(), top_z=float(center[2]))
        c._pour_grounds_once()
        if hasattr(c, "event"):
            c.event("grounds_loaded", counts={k: len(v) for k, v in c.inventory().items()})

    def verify(self):
        c = self.controller
        return bool(c.grounds_ready() and c.holding_loaf and len(c.contacts()) >= 2)


class ParkDosingBoxSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "place_dosing_box")

    def candidates(self):
        return (None,)

    def execute_candidate(self, candidate):
        c = self.controller
        target = np.asarray(c.spec["box_parking_pose"])
        hover = target.copy()
        hover[2, 3] += 0.10
        c.grasp_relative = np.linalg.inv(c.tcp()) @ c.bread_pose()
        c.arm("return empty box above counter", hover @ np.linalg.inv(c.grasp_relative), True)
        c.allow_support = True
        near = target.copy()
        near[2, 3] += 0.003
        c.held_pose("set empty box on counter", near)
        c.release("withdraw from empty box")
        c.event("empty_box_parked")

    def verify(self):
        c = self.controller
        return bool(
            not c.holding_loaf
            and not c.attached
            and not c.contacts()
            and c.assignment([c.spec["dosing_body"]]).get(c.spec["dosing_body"]) == c.counter
            and np.linalg.norm(c.bread_pose()[:3, 3] - np.asarray(c.spec["box_parking_pose"])[:3, 3]) < 0.02
        )


def dosing_sequence(controller):
    return SkillSequence(
        controller,
        "dose_portafilter",
        [
            (PickDosingBoxSkill(controller), {}),
            (PourGroundsSkill(controller, track_portafilter=True), {}),
            (ParkDosingBoxSkill(controller), {}),
        ],
    )


def press_button_skill(controller, action):
    c = controller
    return BoundSkill(
        c,
        "press_" + action + "_button",
        lambda: c.operate_button(action),
        lambda: (
            c.button_pressed
            and not c.button_contact()[0]
            and abs(float(c.data.joint(c.active_button_body.removesuffix("_001")).qpos[0])) < 0.0002
        ),
    )


def coffee_operations(controller, initialize):
    """Single registry shared by the standalone skill runner and all coffee plans."""
    from cross_episode_sim.skills.coffee_cups import PlaceCoffeeCupSkill, RetrieveCoffeeCupSkill
    from cross_episode_sim.skills.composite import Operation
    from cross_episode_sim.skills.portafilter import AttachPortafilterSkill, DetachPortafilterSkill
    from cross_episode_sim.skills.sequence import SequenceOperation

    c = controller
    return dict(
        initialize=Operation(initialize, lambda: not c.inventory()["spilled"]),
        remove=DetachPortafilterSkill(c),
        dose=SequenceOperation(lambda: dosing_sequence(c), lambda: c.grounds_ready() and not c.holding_loaf),
        reinstall=AttachPortafilterSkill(c),
        cup=PlaceCoffeeCupSkill(c),
        power=press_button_skill(c, "power"),
        warmup=Operation(c.wait_machine_ready, lambda: c.machine_cycle.ready(c.data.time)),
        press=press_button_skill(c, "brew"),
        brew=Operation(c.finish_brew, lambda: c.machine_cycle.completed and c.mug_ready()),
        retrieve=RetrieveCoffeeCupSkill(c),
        next_mug=Operation(c.next_mug, lambda: True),
    )
