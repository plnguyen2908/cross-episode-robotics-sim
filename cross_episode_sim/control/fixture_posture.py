"""Empty-hand posture transition after a household fixture interaction.

The scene comes only from onboard depth. Robot calibration supplies the exact
base and gripper narrow phase, as in the fixture approach planner. This is
separate from the qualified room-navigation / set-table implementations.
"""
import time
import numpy as np


def plan_fixture_stow(planner, joints, posture):
    """Try a checked withdrawal waypoint before an expensive graph search."""
    from .contact_retreat import tool_pose
    allow_graph = planner.allow_curobo_fallback
    planner.allow_curobo_fallback = False
    rejected = []
    try:
        try:
            return planner.plan_joint_goal(joints, posture)
        except RuntimeError as exc:
            rejected.append(str(exc))
        tcp = tool_pose(planner.geometry, joints)
        toward_base = -tcp[:3, 3].copy()
        toward_base[2] = 0.
        toward_base /= max(np.linalg.norm(toward_base), 1e-8)
        for distance in (.08, .14, .20):
            for height in (0., -.06, .06):
                waypoint = tcp[:3, 3] + distance * toward_base + [0., 0., height]
                try:
                    first = planner.plan(joints, waypoint, tcp[:3, :3])
                    second = planner.plan_joint_goal(first[-1], posture)
                except RuntimeError as exc:
                    rejected.append(str(exc))
                    continue
                planner.events.append(dict(method='checked_fixture_withdrawal',
                                           waypoint=waypoint.tolist(),
                                           reason='direct stow sweep blocked',
                                           rejected_paths=rejected))
                return np.concatenate([first, second[1:]])
    finally:
        planner.allow_curobo_fallback = allow_graph
    return planner.plan_joint_goal(joints, posture)


def stow_after_fixture(runtime, sensors, recorder, output, posture):
    import mujoco
    from curobo._src.geom.types import Cuboid
    from .sensor_appliance_backend import camera_points
    from .observed_grasp_planner import merge_voxels
    from .observed_arm import robot_calibration
    from .planner_cache import sensor_arm_planner
    from .fixture_approach import calibrated_finger_planner
    from ..navigation.shaped_docking import RobotGeometry
    from ..finetuning.molmospaces_execution import MolmoSpacesController

    started = time.monotonic()
    model, data = runtime.model, runtime.data
    geometry = RobotGeometry(model)
    controller = MolmoSpacesController(model, data)
    q = controller.robot_view.get_move_group('arm').joint_pos.copy()
    target = np.asarray(posture, float)
    packet = sensors.capture()
    clouds = [camera_points(camera)[0] for camera in packet.cameras.values()]
    points = np.concatenate(clouds)
    points = points[(points[:, 2] > .08) & (np.linalg.norm(points, axis=1) < 2.)]
    voxels = np.unique(np.floor(points / .01).astype(int), axis=0)
    boxes = [Cuboid(name=f'fixture_stow_{i}',
                    pose=((lo + hi + 1) * .005).tolist() + [1., 0., 0., 0.],
                    dims=((hi - lo + 1) * .01).tolist())
             for i, (lo, hi) in enumerate(merge_voxels(voxels))]
    calibration = robot_calibration(model, data, output / 'robot')
    arm = sensor_arm_planner(calibration,
                            collision_cache={'cuboid': max(16, len(boxes)), 'mesh': 2},
                            activation_distance=.012)
    planner = calibrated_finger_planner(arm, geometry, boxes, model, data,
                                       allow_curobo_fallback=True)
    print('FIXTURE_STOW_PLAN', dict(boxes=len(boxes)), flush=True)
    path = plan_fixture_stow(planner, q, target)
    hold = {group: controller.robot_view.get_move_group(group).noop_ctrl.copy()
            for group in ('base', 'arm', 'gripper')}
    start_sim = float(data.time)

    def command(joints):
        hold['arm'] = joints
        hold['gripper'] = np.array([0.])
        controller.step(hold)
        for _ in range(round(.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        recorder.record('stow_after_fixture')

    for joints in path:
        command(joints)
    stable = 0
    for _ in range(100):
        command(target)
        measured = controller.robot_view.get_move_group('arm').joint_pos
        velocity = np.array([data.joint(name).qvel[0] for name in geometry.joint_names])
        stable = stable + 1 if (np.max(abs(measured - target)) < .015
                               and np.max(abs(velocity)) < .04) else 0
        if stable >= 5:
            report = dict(execution_success=True,
                          simulated_seconds=float(data.time) - start_sim,
                          wall_seconds=time.monotonic() - started,
                          planning_methods=planner.events,
                          stability_samples=stable,
                          scope='sensor-only empty-hand fixture posture transition')
            print('FIXTURE_STOW_COMPLETE', report, flush=True)
            return report
    raise RuntimeError('Fixture posture did not settle from arm proprioception')
