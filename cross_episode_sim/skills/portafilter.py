"""Shared physical bayonet detach/attach tasks, used by demos and composites."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cross_episode_sim.control.collision_probe import forward_collision
from cross_episode_sim.manipulation.edge_access import object_bodies
from cross_episode_sim.skills.atomic import AtomicSkill
from cross_episode_sim.tasks.coffee.machine import PORTAFILTER


def body_pose(data, name):
    body = data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def twist_angles(step, *, locking=False):
    """Larger servo targets, with the original 50-degree endpoints preserved."""
    if step not in (5, 10, 15, 25):
        raise ValueError('Portafilter twist increment must be 5, 10, 15, or 25 degrees')
    progress = [*range(step, 50, step), 50]
    return [50-angle for angle in progress] if locking else progress


def bayonet_waypoints(start, target, object_T_tcp):
    """Follow the basket's rotation arc, rather than a chord between TCP poses."""
    angle = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude()
    count = max(6, int(np.ceil(angle / np.deg2rad(1.5))))
    rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
    result = []
    for fraction in np.linspace(0, 1, count+1)[1:]:
        pose = np.eye(4)
        pose[:3, :3] = rotations(fraction).as_matrix()
        pose[:3, 3] = (1-fraction)*start[:3, 3] + fraction*target[:3, 3]
        result.append((pose, pose @ object_T_tcp))
    return result


def twist_to(c, stage, target):
    if c.spec.get('portafilter_twist_step_deg', 5) == 5:
        return c.held_pose(stage, target)
    start = c.bread_pose()
    object_T_tcp = np.linalg.inv(start) @ c.tcp()
    addresses = [c.model.jnt_qposadr[c.model.joint(c.profile.namespace+n).id] for n in c.planner.names]
    address = c.model.jnt_qposadr[c.model.joint(c.object_joint).id]
    probe = mujoco.MjData(c.model)
    q, path = c.q(), []
    for body_target, tcp_target in bayonet_waypoints(start, target, object_T_tcp):
        q = np.asarray(c.nearby_ik(tcp_target, q, preserve_self_clearance=True))
        probe.qpos[:] = c.data.qpos
        probe.qpos[addresses] = q
        probe.qpos[address:address+3] = body_target[:3, 3]
        probe.qpos[address+3:address+7] = Rotation.from_matrix(body_target[:3, :3]).as_quat(scalar_first=True)
        forward_collision(c.model, probe)
        if c.navigation_penetration(probe, True) > .001 or c.robot_self_penetration(probe) > .0005:
            raise RuntimeError(f'Actual-mesh collision in bayonet arc: {stage}; {c.contact_pair_detail(probe, True)}')
        path.append(q.copy())
    c.record(bayonet_arc_waypoints=len(path), bayonet_arc_max_increment_deg=1.5)
    # The same joint feedback servo, contact monitoring, and measured payload
    # correction apply. Only the checked path follows the socket's actual arc.
    return c.held_pose(stage, target, path=path)


def detached(controller):
    c = controller
    pose = body_pose(c.data, PORTAFILTER)
    own = object_bodies(c.model, PORTAFILTER)
    support = c.table_bids[c.counter]
    supported = any(
        (c.model.geom_bodyid[x.geom1] in own and c.model.geom_bodyid[x.geom2] in support)
        or (c.model.geom_bodyid[x.geom2] in own and c.model.geom_bodyid[x.geom1] in support)
        for x in c.data.contact
    )
    return bool(
        not c.data.eq_active[c.lock_eq]
        and not np.any(c.data.eq_active[c.axis_eq])
        and supported
        and np.linalg.norm(pose[:3, 3] - c.loading[:3, 3]) < 0.015
        and Rotation.from_matrix(pose[:3, :3] @ c.loading[:3, :3].T).magnitude() < np.deg2rad(5)
        and np.linalg.norm(c.data.joint(PORTAFILTER + "_free").qvel) < 0.01
        and not c.holding_loaf
        and not c.attached
        and not c.contacts()
    )


class PortafilterSkill(AtomicSkill):
    def candidates(self):
        # One validated physical path. Do not repeat an identical failed attempt.
        return (None,)

    def rebuild(self):
        self.controller.rebuild()


class DetachPortafilterSkill(PortafilterSkill):
    def __init__(self, controller):
        super().__init__(controller, "detach_portafilter")

    def verify(self):
        return detached(self.controller)

    def execute_candidate(self, candidate):
        c = self.controller
        c.review_phase = "REMOVE PORTAFILTER"
        c.pickup(PORTAFILTER, c.pf_grasp())
        # A previously attached basket has its seated lock active. Only a
        # measured grasp may change it to the axis-guided unlocking mode.
        if len(c.contacts()) < 2:
            raise RuntimeError("Portafilter unlock requires bilateral handle contact")
        if c.data.eq_active[c.lock_eq]:
            c.data.eq_active[c.axis_eq] = True
            c.data.eq_active[c.lock_eq] = False
            c.event("bayonet_unlock_started")
        angles = twist_angles(c.spec.get('portafilter_twist_step_deg', 5))
        c.report['portafilter_unlock_angles_deg'] = angles
        for angle in angles:
            target = c.seated.copy()
            target[:3, :3] = Rotation.from_euler("z", -angle, degrees=True).as_matrix() @ c.seated[:3, :3]
            twist_to(c, "twist portafilter to unlock", target)
        actual = body_pose(c.data, PORTAFILTER)
        error = Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()
        if error > np.deg2rad(3):
            raise RuntimeError("Portafilter did not physically unlock")
        c.data.eq_active[c.axis_eq] = False
        c.fixture_phase = "released"
        c.event("bayonet_unlocked", orientation_error_deg=float(np.rad2deg(error)))
        down = actual.copy()
        for amount in (0.02, 0.04, 0.06, 0.08):
            p = down.copy()
            p[2, 3] -= amount
            c.held_pose("lower unlocked portafilter", p)
        front = c.toward_loading(p)
        c.held_pose("withdraw portafilter from machine", front)
        hover = c.loading.copy()
        hover[2, 3] += 0.012
        c.held_pose("position portafilter above counter", hover)
        c.held_pose("lower portafilter onto counter", c.loading)
        c.portafilter_pick_seed = c.q().tolist()
        c.remember_removal(c.trace)
        c.release("withdraw empty hand from portafilter")
        c.tick(0.8)
        if np.linalg.norm(body_pose(c.data, PORTAFILTER)[:3, 3] - c.loading[:3, 3]) > 0.015:
            raise RuntimeError("Portafilter unstable on counter")
        own = object_bodies(c.model, PORTAFILTER)
        counter = c.table_bids[c.counter]
        supported = any(
            (c.model.geom_bodyid[contact.geom1] in own and c.model.geom_bodyid[contact.geom2] in counter)
            or (c.model.geom_bodyid[contact.geom2] in own and c.model.geom_bodyid[contact.geom1] in counter)
            for contact in c.data.contact
        )
        tilt = Rotation.from_matrix(body_pose(c.data, PORTAFILTER)[:3, :3] @ c.loading[:3, :3].T).magnitude()
        speed = float(np.linalg.norm(c.data.joint(PORTAFILTER + "_free").qvel))
        if not supported or tilt > np.deg2rad(5) or speed > 0.01:
            raise RuntimeError("Portafilter lacks stable upright counter support")
        c.event(
            "portafilter_removed", pose=body_pose(c.data, PORTAFILTER).tolist(), counter_contact=supported, speed=speed
        )


class AttachPortafilterSkill(PortafilterSkill):
    def __init__(self, controller):
        super().__init__(controller, "attach_portafilter")

    def verify(self):
        return bool(
            self.controller.installed()
            and not self.controller.holding_loaf
            and not self.controller.attached
            and not self.controller.contacts()
        )

    def execute_candidate(self, candidate):
        c = self.controller
        c.review_phase = "REINSTALL LOADED PORTAFILTER"
        c.pickup(PORTAFILTER, c.pf_grasp())
        lift = c.bread_pose().copy()
        lift[2, 3] += 0.025
        c.held_pose("lift loaded portafilter", lift)
        c.allow_support = False
        target = c.seated.copy()
        target[:3, :3] = Rotation.from_euler("z", -50, degrees=True).as_matrix() @ c.seated[:3, :3]
        low = target.copy()
        low[2, 3] -= 0.08
        front = c.toward_loading(low)
        c.held_pose("align loaded portafilter in front of machine", front)
        c.held_pose("insert loaded portafilter below socket", low)
        for amount in (0.06, 0.04, 0.02, 0.0):
            p = target.copy()
            p[2, 3] -= amount
            c.held_pose("raise portafilter into socket", p)
        actual = body_pose(c.data, PORTAFILTER)
        if np.linalg.norm(actual[:3, 3] - target[:3, 3]) > 0.002 or Rotation.from_matrix(
            actual[:3, :3] @ target[:3, :3].T
        ).magnitude() > np.deg2rad(2):
            raise RuntimeError("Portafilter outside bayonet engagement tolerance")
        c.data.eq_active[c.axis_eq] = True
        c.event("bayonet_engaged", position_error_m=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])))
        angles = twist_angles(c.spec.get('portafilter_twist_step_deg', 5), locking=True)
        c.report['portafilter_lock_angles_deg'] = angles
        for angle in angles:
            p = c.seated.copy()
            p[:3, :3] = Rotation.from_euler("z", -angle, degrees=True).as_matrix() @ c.seated[:3, :3]
            twist_to(c, "twist portafilter to lock", p)
        actual = body_pose(c.data, PORTAFILTER)
        if np.linalg.norm(actual[:3, 3] - c.seated[:3, 3]) > 0.002 or Rotation.from_matrix(
            actual[:3, :3] @ c.seated[:3, :3].T
        ).magnitude() > np.deg2rad(2):
            raise RuntimeError("Portafilter did not reach locked pose")
        c.data.eq_active[c.lock_eq] = True
        c.data.eq_active[c.axis_eq] = False
        c.fixture_phase = "locked"
        c.release("withdraw hand from installed portafilter")
        c.tick(0.5)
        if not c.installed():
            raise RuntimeError("Portafilter installation failed")
        c.event("portafilter_reinstalled", counts={k: len(v) for k, v in c.inventory().items()})
