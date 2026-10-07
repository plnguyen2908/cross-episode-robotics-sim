"""Bounded, state-dependent settling for continuous physical execution."""

import numpy as np


def wait_until_ready(controller, budget, ready, *, label, after_step=None):
    """No mandatory hold when ready; never skip physics or hide recorded steps.

    Legacy tasks retain their original timing until explicitly enabled. The
    budget is the former fixed wait, not a new or extended timeout. Existing
    endpoint/support validators remain responsible for rejecting failed motion.
    """
    if not getattr(controller, "adaptive_motion_settling", False):
        if after_step is None:
            controller.tick(budget)
        else:
            elapsed = 0.
            while elapsed < budget - 1e-9:
                step = min(.04, budget - elapsed)
                controller.tick(step)
                after_step()
                elapsed += step
        return
    start = float(controller.data.time)
    satisfied = bool(ready())
    while not satisfied and controller.data.time - start < budget - 1e-9:
        controller.tick(min(.02, budget - (controller.data.time - start)))
        if after_step is not None:
            after_step()
        satisfied = bool(ready())
    elapsed = float(controller.data.time) - start
    stats = controller.report.setdefault("motion_settling", {})
    item = stats.setdefault(label, dict(calls=0, elapsed_s=0., saved_s=0., budget_exhausted=0))
    item["calls"] += 1
    item["elapsed_s"] += elapsed
    item["saved_s"] += max(0., budget - elapsed)
    item["budget_exhausted"] += int(not satisfied)


def payload_ready(controller, linear=.01, angular=.05):
    """A held object can keep swinging in the grip after the arm has stopped."""
    joint = getattr(controller, "object_joint", None)
    if not getattr(controller, "holding_loaf", False) or joint is None:
        return True
    velocity = np.asarray(controller.data.joint(joint).qvel)
    if velocity.size != 6:
        return True
    return bool(np.linalg.norm(velocity[:3]) <= linear and np.linalg.norm(velocity[3:]) <= angular)


def arm_ready(controller, pose=None, position_tolerance=.005):
    names = controller.planner.names
    joints = [controller.data.joint(controller.profile.namespace + n) for n in names]
    # A held object is released right after this kind of move; residual hand
    # motion at release can knock it, so loaded moves settle more completely.
    holding = getattr(controller, "holding_loaf", False)
    if any(abs(float(j.qvel[0])) > (.005 if holding else .02) for j in joints):
        return False
    if not payload_ready(controller):
        return False
    if pose is not None:
        current = controller.tcp()
        rotation_error = np.arccos(np.clip(
            (np.trace(current[:3, :3].T @ pose[:3, :3]) - 1.) / 2., -1., 1.))
        return (np.linalg.norm(current[:3, 3] - pose[:3, 3]) <= position_tolerance
                and rotation_error <= .02)
    actual = np.array([float(j.qpos[0]) for j in joints])
    return bool(np.max(np.abs(controller.data.ctrl[controller.arm_aids] - actual)) <= .02)


def settle_arm(controller, budget, pose=None, position_tolerance=.005):
    return wait_until_ready(controller, budget,
        lambda: arm_ready(controller, pose, position_tolerance), label="arm")


def gripper_open_ready(controller):
    aid = controller.model.actuator(
        controller.profile.namespace + controller.profile.gripper_actuator).id
    kp = float(controller.model.actuator_gainprm[aid, 0])
    if kp <= 0:
        return False
    # Use the actual servo transmission rather than assuming finger coordinates
    # equal control units (Franka and RB-Y1 use different transmissions).
    scale = -float(controller.model.actuator_biasprm[aid, 1]) / kp
    span = float(np.ptp(controller.model.actuator_ctrlrange[aid]))
    position_error = abs(float(controller.data.ctrl[aid])
                         - scale * float(controller.data.actuator_length[aid]))
    speed = abs(scale * float(controller.data.actuator_velocity[aid]))
    return position_error <= .02 * span and speed <= .02 * span


def wait_for_open_gripper(controller, budget):
    return wait_until_ready(controller, budget,
        lambda: gripper_open_ready(controller), label="open_gripper")


def settle_released_object(controller, budget):
    def ready():
        velocity = controller.data.joint(controller.object_joint).qvel
        if not (gripper_open_ready(controller)
                and np.linalg.norm(velocity[:3]) <= .01
                and np.linalg.norm(velocity[3:]) <= .05):
            return False
        try:
            return controller.assignment()[controller.object_name] == controller.destination
        except RuntimeError as exc:
            # A just-released vessel can briefly have no support. Poll until
            # contact forms; the caller still rejects missing support at timeout.
            if str(exc).startswith('Missing or ambiguous native table support:'):
                return False
            raise
    return wait_until_ready(controller, budget, ready, label="release")


def base_ready(controller, target):
    delta = controller.base_pose() - target
    velocity = [float(controller.data.joint(controller.profile.namespace + n).qvel[0])
                for n in controller.profile.base_joints]
    return (np.linalg.norm(delta[:2]) <= .002 and abs(delta[2]) <= np.radians(.2)
            and np.linalg.norm(velocity[:2]) <= .005 and abs(velocity[2]) <= .01)
