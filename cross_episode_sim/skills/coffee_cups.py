"""Coffee-cup pick/place variants for constrained spout access."""

import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.edge_access import object_bodies
from cross_episode_sim.skills.atomic import AtomicSkill
from cross_episode_sim.skills.portafilter import body_pose
from cross_episode_sim.tasks.coffee.placement import rotate, to_validated, to_world

TRAY = "drip_tray_grate_lift_pivot_001"
PUSH_DOWN = 0.0


class PlaceCupAtSpoutSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "place_cup")

    def candidates(self):
        return (.001, .0005, .0015)

    def verify(self):
        c = self.controller
        return bool(c.mug_ready() and not c.attached and not c.holding_loaf and not c.contacts())

    def execute_candidate(self, candidate):
        c = self.controller
        c.review_phase = "PLACE CUP UNDER SPOUT"
        c.extra_support_bids = object_bodies(c.model, TRAY) | object_bodies(c.model, "drip_tray_slide_pivot_001")
        c.allow_support = False
        target = np.asarray(c.spec["mug_target_pose"])
        # The native mug nearly fills the gap. Enter beside the low forward
        # handle, then slide laterally below the basket; avoid tilting its rim.
        target = target.copy()
        target[2, 3] -= 0.0003
        clearance = .001 if candidate is None else float(candidate)
        c.record(spout_insertion_clearance_m=clearance)
        # Waypoints are set in the validated layout and mapped to the placement.
        local = to_validated(c.spec, target)
        front_local = local.copy()
        # Sliding at the final seated height loads the tray edge and can tip
        # the cup while the arm servo pushes forward. Clear it during insertion;
        # lower only after the cup is centered under the basket.
        front_local[2, 3] += clearance
        front_local[:2, 3] = [2.20, -0.46]
        front = to_world(c.spec, front_local)
        c.grasp_relative = np.linalg.inv(c.tcp()) @ c.bread_pose()
        c.arm("approach beside portafilter handle", front @ np.linalg.inv(c.grasp_relative), True)
        c.held_pose("level cup beside machine", front)
        c.allow_support = True
        side_local = front_local.copy()
        side_local[1, 3] = -0.31
        side = to_world(c.spec, side_local)
        self.insert("insert cup beside portafilter handle", side)
        under_local = side_local.copy()
        under_local[0, 3] = local[0, 3]
        under = to_world(c.spec, under_local)
        self.insert("slide cup below portafilter basket", under)
        centered = target.copy()
        centered[2, 3] += clearance
        self.insert("center cup under coffee spout", centered)
        c.held_pose("lower cup onto drip tray", target)
        c.release("withdraw hand from coffee cup", retreat_delta=rotate(c.spec, np.eye(3)) @ [0.04, -0.06, 0.0])
        c.tick(0.5)
        # Where the retreat started, relative to the settled mug: retrieval retraces it.
        c.__dict__.setdefault("mug_release_tcp", {})[c.mug] = (
            np.linalg.inv(body_pose(c.data, c.mug)) @ c.last_release_tcp
        ).tolist()
        c.extra_support_bids = set()
        if not c.mug_ready():
            raise RuntimeError("Cup is not upright, supported and aligned under spout")
        c.event("cup_placed", pose=c.bread_pose().tolist())

    def insert(self, stage, target):
        """Recheck the cup pose during insertion, before tilt reaches the tray.

        Joint tracking alone does not detect a cup rotating within the fingers.
        Each short segment remeasures the grasp transform and applies the same
        1 mm / 0.5 degree payload checks as the final target.
        """
        c = self.controller
        start = c.bread_pose().copy()
        count = max(1, int(np.ceil(np.linalg.norm(target[:3, 3]-start[:3, 3])/.03)))
        for fraction in np.linspace(0., 1., count+1)[1:]:
            waypoint = target.copy()
            waypoint[:3, 3] = start[:3, 3] + fraction*(target[:3, 3]-start[:3, 3])
            c.held_pose(stage, waypoint)


class PickCupFromSpoutSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "pick_cup_from_spout")

    def candidates(self):
        # Small lateral extraction alternatives move the gripper linkage clear
        # of the handle. Each is checked by the unchanged actual-mesh limits.
        return (0., .004, .008, -.004)

    def verify(self):
        c = self.controller
        return bool(c.object_name == c.mug and c.holding_loaf and len(c.contacts()) >= 2)

    def execute_candidate(self, candidate):
        """Grasp and extract the mug beside the handle, leaving it held."""
        c = self.controller
        mug = c.mug
        c.review_phase = "RETRIEVE CUP FROM SPOUT"
        c.record(spout_extraction_offset_m=0. if candidate is None else float(candidate))
        c.extra_support_bids = object_bodies(c.model, TRAY) | object_bodies(c.model, "drip_tray_slide_pivot_001")
        # Grasp where the hand released the mug and approach along that release's
        # retreat, which was collision-checked beside the portafilter handle.
        local = np.array(c.mug_release_tcp[mug])
        grasp = body_pose(c.data, mug) @ local
        pre = grasp.copy()
        pre[:3, 3] += rotate(c.spec, np.eye(3)) @ [0.04, -0.06, 0.0]
        # Fully open (85 mm) the Robotiq linkage spreads 2 mm into the portafilter
        # handle at this pose. The rim grasp spans 21 mm, so approach about 42 mm
        # open (command 130 of 0-255 closed), about 10 mm clear on each side.
        c.pickup(mug, local, pre=pre, approach_command=130.0)
        # Retrace placement's own waypoints: they come from the computed target
        # pose 0.3 mm into the tray, not from where the mug settled on the grate
        # (about 0.5 mm higher, enough to catch the basket in this tight gap).
        planned = np.asarray(c.spec["mug_target_pose"]).copy()
        planned[2, 3] -= 0.0003
        local_target = to_validated(c.spec, planned)
        front_local = local_target.copy()
        front_local[:2, 3] = [2.20, -0.46]
        # Slide out pressed down onto the drip tray: the rim then stays below the
        # basket, and the cup may rub the machine on its way out.
        side_local = front_local.copy()
        side_local[1, 3] = -0.31
        side_local[2, 3] -= PUSH_DOWN
        under_local = side_local.copy()
        under_local[0, 3] = local_target[0, 3] + (0. if candidate is None else candidate)
        c.payload_tolerance = (PUSH_DOWN + 0.0015, 1.5)
        c.payload_contact_ok_bids = object_bodies(c.model, "base_frame_002") | object_bodies(
            c.model, "portafilter_extract_pivot_001"
        )
        try:
            c.held_pose("slide cup out below portafilter basket", to_world(c.spec, under_local))
            c.held_pose("withdraw cup beside portafilter handle", to_world(c.spec, side_local))
            c.allow_support = False
            c.held_pose("level cup beside machine", to_world(c.spec, front_local))
        finally:
            c.payload_tolerance = None
            c.payload_contact_ok_bids = set()
        # Extraction frees the machine regardless of whether this cup goes home
        # or straight to its dining setting. Keep power/warmup and filled-cup history.
        c.machine_cycle.next_cup()
        c.event('cup_extracted', mug=mug, filled=mug in c.coffee_filled)
class PlaceCupOnCounterSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "place_cup_on_counter")

    def candidates(self):
        return (None,)

    def verify(self):
        c = self.controller
        return bool(
            not c.holding_loaf
            and not c.attached
            and not c.contacts()
            and c.machine_cycle.started_at is None
            and np.linalg.norm(body_pose(c.data, c.mug)[:3, 3] - np.asarray(c.mug_homes[c.mug])[:3, 3]) < 0.02
            and c.data.body(c.mug).xmat[8] > 0.995
        )

    def execute_candidate(self, candidate):
        c = self.controller
        mug = c.mug
        info = next(i for i in c.manifest['bindings'] if i['body'] == mug)
        c.extra_support_bids = object_bodies(c.model, c.task_supports[info['source']])
        home = np.array(c.mug_homes[mug])
        hover = home.copy()
        hover[2, 3] += 0.07
        c.grasp_relative = np.linalg.inv(c.tcp()) @ c.bread_pose()
        c.arm("carry brewed cup to counter spot", hover @ np.linalg.inv(c.grasp_relative), True)
        c.allow_support = True
        near = home.copy()
        near[2, 3] += 0.003
        c.held_pose("set brewed cup on counter", near)
        c.release("withdraw hand from brewed cup")
        c.tick(0.5)
        c.extra_support_bids = set()
        if np.linalg.norm(body_pose(c.data, mug)[:3, 3] - home[:3, 3]) > 0.02 or c.data.body(mug).xmat[8] < 0.995:
            raise RuntimeError("Brewed cup is not upright at its counter spot")
        c.machine_cycle.next_cup()
        c.event("cup_retrieved", mug=mug, filled=mug in c.coffee_filled, pose=body_pose(c.data, mug).tolist())


class PlaceCupAtSettingSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "place_cup_at_setting")

    def candidates(self, role, support):
        return (None,)

    def verify(self, role, support):
        return bool(self.controller.verify_role(role))

    def execute_candidate(self, candidate, role, support):
        """Set the held cup down from above at the current dining dock."""
        c = self.controller
        info = c.info(role)
        body = info["body"]
        c.active_mug = body
        target = np.asarray(info["serving_position"], dtype=float)
        # Upright at the place setting with the handle (mug +x, where the hand
        # holds it) toward the robot, so the hand stays within easy reach.
        base = c.base_pose()
        yaw = float(np.arctan2(base[1] - target[1], base[0] - target[0]))
        place = np.eye(4)
        place[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        place[:3, 3] = target
        hover = place.copy()
        hover[2, 3] += 0.07
        table = c.task_supports["dining"]
        c.extra_support_bids = object_bodies(c.model, table)
        c.grasp_relative = np.linalg.inv(c.tcp()) @ c.bread_pose()
        c.arm("carry cup above its place setting", hover @ np.linalg.inv(c.grasp_relative), True)
        c.allow_support = True
        near = place.copy()
        near[2, 3] += 0.003
        c.held_pose("set cup at its place setting", near)
        c.release("withdraw hand from served cup")
        c.tick(0.5)
        c.extra_support_bids = set()
        c.report.setdefault("executed_placement_targets", {})[role] = target.tolist()


class PickCounterCupSkill(AtomicSkill):
    def __init__(self, controller):
        super().__init__(controller, "pick_cup_from_counter")

    def candidates(self):
        return (None,)

    def verify(self):
        c = self.controller
        return bool(c.object_name == c.mug and c.holding_loaf and len(c.contacts()) >= 2)

    def execute_candidate(self, candidate):
        c = self.controller
        local = np.load(next(i for i in c.manifest['bindings'] if i['body'] == c.mug)['grasp_path'])["transforms"][c.spec.get("coffee_mug_grasp_index", 10)]
        c.__dict__.setdefault("mug_homes", {}).setdefault(c.mug, body_pose(c.data, c.mug).tolist())
        info = next(i for i in c.manifest['bindings'] if i['body'] == c.mug)
        c.extra_support_bids = object_bodies(c.model, c.task_supports[info['source']])
        c.pickup(c.mug, local)
        lift = c.bread_pose().copy()
        lift[2, 3] += 0.07
        c.held_pose("lift coffee cup", lift)


class _CupRecipe:
    """Legacy scripted-history adapter, explicitly a recipe, never an atom."""
    def __init__(self, controller):
        self.controller = controller

    def validate_arguments(self, **arguments):
        import inspect
        inspect.signature(self.run).bind(**arguments)


class PlaceCoffeeCupSkill(_CupRecipe):
    def verify(self):
        return PlaceCupAtSpoutSkill(self.controller).verify()

    def run(self):
        c = self.controller
        if c.spec.get("navigate_to_mugs", False):
            c.walk_to_counter(c.mug, body_pose(c.data, c.mug)[:3, 3])
        PickCounterCupSkill(c).run()
        if c.spec.get("navigate_to_mugs", False):
            c.return_to_machine()
        PlaceCupAtSpoutSkill(c).run()


class RetrieveCoffeeCupSkill(_CupRecipe):
    def verify(self):
        return PlaceCupOnCounterSkill(self.controller).verify()

    def run(self):
        c = self.controller
        PickCupFromSpoutSkill(c).run()
        if c.spec.get("navigate_to_mugs", False):
            c.walk_to_counter(c.mug, np.asarray(c.mug_homes[c.mug])[:3, 3], carrying=True)
        PlaceCupOnCounterSkill(c).run()


class ServeCoffeeCupSkill(_CupRecipe):
    def verify(self, role, support):
        return PlaceCupAtSettingSkill(self.controller).verify(role, support)

    def run(self, role, support):
        c = self.controller
        c.active_mug = c.info(role)["body"]
        c.walk_to_counter(c.mug, c.data.body(c.mug).xpos.copy())
        PickCounterCupSkill(c).run()
        c.walk_to_counter(c.mug, np.asarray(c.info(role)["serving_position"]), carrying=True, room="dining")
        PlaceCupAtSettingSkill(c).run(role=role, support=support)
