"""Physical coffee button skills shared by standalone and composite tasks."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.manipulation.edge_access import collision_vertices, object_bodies
from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.tasks.coffee.placement import to_validated, to_world


def press_coffee_button(c):
    action = getattr(c, "machine_action", None) or "start"
    c.review_phase = "PRESS MACHINE " + action.upper() + " BUTTON"
    c.embodiment.command_gripper(c.data, c.profile.gripper_close)
    c.tick(0.5)
    c.press_phase = True
    c.rebuild()
    pad = object_bodies(c.model, c.profile.namespace + "gripper/left_pad")
    pad_local = (collision_vertices(c.model, c.data, pad) - c.tcp()[:3, 3]) @ c.tcp()[:3, :3]
    # Plan the press in the validated layout (button face toward -y), then
    # map every pose into the machine's actual placement.
    points = collision_vertices(c.model, c.data, object_bodies(c.model, c.active_button_body))
    points = np.array([to_validated(c.spec, point) for point in points])
    surface = (points.min(0) + points.max(0)) / 2
    surface[1] = points[:, 1].min()
    addresses = [c.model.jnt_qposadr[c.model.joint(c.profile.namespace + n).id] for n in c.planner.names]
    probe = mujoco.MjData(c.model)
    selected = None
    seeds = [np.array([0.0, -0.3, 0.0, -1.6, 0.0, 1.8, 0.0]), c.q(), np.array([0.0, -0.3, 0.0, -1.6, 0.0, 1.8, 2.5])]
    for tilt in (-20.0, 35.0, 0.0, 60.0):
        R = Rotation.from_euler("x", tilt, degrees=True).as_matrix() @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        offsets = pad_local @ R.T
        tip = offsets[offsets[:, 1] > offsets[:, 1].max() - 1e-5].mean(0)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = surface - tip
        pre = pose.copy()
        pre[1, 3] -= 0.075
        for seed in seeds:
            try:
                endpoint = c.nearby_ik(to_world(c.spec, pre), seed, trust_radius=6.0, preserve_self_clearance=True)
                q = endpoint
                for offset in np.linspace(-0.075, 0.0015, 17):
                    target = pose.copy()
                    target[1, 3] += offset
                    q = c.nearby_ik(to_world(c.spec, target), q, preserve_self_clearance=True)
                    probe.qpos[:] = c.data.qpos
                    probe.qpos[addresses] = q
                    mujoco.mj_forward(c.model, probe)
                    if c.navigation_penetration(probe, False) > 0.001 or c.robot_self_penetration(probe) > 0.0005:
                        raise RuntimeError(
                            f"Button approach actual-mesh collision at {offset}: {c.contact_pair_detail(probe, False)}"
                        )
                path = c.planner.plan_joints(c.q(), endpoint)
                selected = (pose, pre, path)
                break
            except RuntimeError as exc:
                c.record(button_approach_rejection=dict(tilt_deg=tilt, error=str(exc)))
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("No checked arm posture for complete button press")
    pose, pre, path = selected
    c.record(button_approach_tilt_deg=tilt)
    c.preplanned_moves["approach coffee button"] = (to_world(c.spec, pre), path)
    c.arm("approach coffee button", to_world(c.spec, pre), True)
    for offset in (-0.025, -0.01, -0.003, 0.0, 0.0005, 0.001, 0.0015):
        target = pose.copy()
        target[1, 3] += offset
        c.arm("press coffee button", to_world(c.spec, target))
        c.tick(0.15)
        if c.button_pressed:
            break
    else:
        raise RuntimeError("No measured passive button actuation")
    c.arm("withdraw from coffee button", to_world(c.spec, pre))
    c.tick(0.5)
    c.press_phase = False
    if c.button_contact()[0]:
        raise RuntimeError("Finger remains on button")
    c.event(
        "button_released",
        button=c.active_button_body,
        travel_m=float(c.data.joint(c.active_button_body.removesuffix("_001")).qpos[0]),
    )


def press_native_coffee_button(c):
    if not (c.grounds_ready() and c.lid_closed() and c.mug_ready()):
        raise RuntimeError("Coffee Start requires grounds, closed lid and correctly positioned mug")
    c.tuck_for_navigation()
    c.navigate_to_site("coffee_machine", c.spec["dispenser_target"], False)
    c.embodiment.command_gripper(c.data, c.profile.gripper_close)
    c.tick(0.5)
    c.untuck_for_manipulation()
    c.planner = c.make_planner()
    c.arm_aids = c.actuator_ids(c.planner.names)
    c.press_phase = True
    c.load_world()
    R = Rotation.from_euler("x", 35, degrees=True).as_matrix() @ np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    pad = object_bodies(c.model, c.profile.namespace + "gripper/left_pad")
    offsets = (collision_vertices(c.model, c.data, pad) - c.tcp()[:3, 3]) @ c.tcp()[:3, :3] @ R.T
    tip = offsets[offsets[:, 1] > offsets[:, 1].max() - 1e-5].mean(0)
    gid = max(c.button_ids, key=lambda g: c.data.geom_xpos[g, 2])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = c.data.geom_xpos[gid] + (corners * c.model.geom_size[gid]) @ c.data.geom_xmat[gid].reshape(3, 3).T
    surface = c.data.geom_xpos[gid].copy()
    surface[1] = vertices[:, 1].min()
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = surface - tip
    pre = pose.copy()
    pre[1, 3] -= 0.07
    c.move("approach coffee Start", pre)
    for offset in (-0.025, -0.01, -0.003, -0.001, 0.0, 0.0005):
        goal = pose.copy()
        goal[1, 3] += offset
        c.move("press coffee Start", goal)
        c.tick(0.15)
        if c.brew.started_at is not None:
            break
    else:
        raise RuntimeError("Coffee Start did not register a force-bearing physical press")
    c.move("withdraw from coffee Start", pre)
    c.tick(0.4)
    c.press_phase = False
    c.load_world()
    c.tuck_for_navigation()
    if c.button_contact()[0]:
        raise RuntimeError("Gripper still contacts coffee Start after withdrawal")


def native_coffee_button_skill(controller):
    c = controller
    return BoundSkill(
        c,
        "press_coffee_start",
        lambda: press_native_coffee_button(c),
        lambda: c.brew.started_at is not None and not c.button_contact()[0],
    )
