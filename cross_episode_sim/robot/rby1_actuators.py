"""Match RB-Y1's finger actuators to its position-command controller."""

import mujoco


def configure_rby1_gripper_servos(spec: mujoco.MjSpec, namespace: str = "robot_0/") -> None:
    """Convert shipped force motors to bounded position servos, in memory.

    Both RBY1 gripper move groups accept metres in [-0.05, 0]. Passing those
    values directly to the asset's force motors produces almost no force and
    zero closing force. Preserve assets that already declare affine servos.
    RobotConfig gain/force overrides may still be applied after this function.
    """
    for side in ("right", "left"):
        actuator = spec.actuator(f"{namespace}{side}_finger_act")
        if actuator is None:
            raise ValueError(f"Missing RB-Y1 {side} finger actuator")
        if actuator.biastype != mujoco.mjtBias.mjBIAS_NONE:
            continue
        actuator.gainprm[0] = 1000.0
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[:3] = [0.0, -1000.0, -10.0]
        actuator.ctrllimited = True
        actuator.ctrlrange = [-0.05, 0.0]
        actuator.forcelimited = True
        actuator.forcerange = [-20.0, 20.0]
