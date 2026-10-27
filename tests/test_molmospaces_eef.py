"""Native pose-controller checks using a saved scene, without changing physics."""
from pathlib import Path
import mujoco
import numpy as np
import pytest

SOURCE=Path('runs/train_test_v1/train/batches/round_0162/set_table_000162/action_recording')


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local successful source')
def test_native_pose_frame_and_no_state_reset(monkeypatch):
    from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController
    m=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'));d=mujoco.MjData(m)
    z=np.load(SOURCE/'initial_state.npz');mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
    mujoco.mj_forward(m,d)
    controller=MolmoSpacesPoseController(m,d)
    view=controller.joints.robot_view
    pose=controller.kinematics.fk(controller.seed,np.eye(4),rel_to_base=True)['gripper'].copy()
    physical=view.get_move_group('gripper').leaf_frame_to_robot
    np.testing.assert_allclose(pose,physical,atol=1e-9)
    live_arm=view.get_move_group('arm').joint_pos.copy()
    controller.seed['arm']+=.25  # A stale commanded posture must not seed IK.
    seeds=[];native_ik=controller.kinematics.ik
    def capture_seed(*args,**kwargs):
        seeds.append(args[3]['arm'].copy())
        return native_ik(*args,**kwargs)
    monkeypatch.setattr(controller.kinematics,'ik',capture_seed)
    before=d.qpos.copy();time=d.time
    controller.step({'base':view.get_move_group('base').noop_ctrl,'eef_pose':pose,
                     'gripper':view.get_move_group('gripper').noop_ctrl})
    np.testing.assert_array_equal(d.qpos,before)
    assert d.time==time
    assert controller.max_pose_error<1e-4
    assert np.max(np.abs(seeds[0]-live_arm))<=.0200001
    previous_ctrl=d.ctrl.copy();previous_calls=controller.ik_calls
    with pytest.raises(ValueError,match='base'):
        controller.step({'base':[np.nan,0,0],'eef_pose':pose,'gripper':[0.]})
    np.testing.assert_array_equal(d.ctrl,previous_ctrl)
    assert controller.ik_calls==previous_calls
    with pytest.raises(ValueError,match='rigid pose'):
        bad=pose.copy();bad[0,0]*=2
        controller.step({'base':np.zeros(3),'eef_pose':bad,'gripper':[0.]})


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local successful source')
def test_reachable_pose_recovers_from_wrist_limit_without_joint_hint():
    from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController
    m=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'));d=mujoco.MjData(m)
    z=np.load(SOURCE/'initial_state.npz');mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
    mujoco.mj_forward(m,d)
    controller=MolmoSpacesPoseController(m,d)
    # Regression: near-straight elbow branch from the failed pregrasp.
    controller.seed['arm']=np.array([.04818877,-.03695476,.06572835,-.22589435,
                                    -.11410922,.54905946,-.37650583])
    # Set up the measured regression posture in this unit-test fixture only.
    controller.joints.robot_view.get_move_group('arm').joint_pos=controller.seed['arm']
    mujoco.mj_forward(m,d)
    pose=np.array([[.8626297253578729,.36225513264502507,.35304557184779006,.23572063844332966],
                   [.3950512641164863,-.9183726918200971,-.022936818416513932,.002856019440933576],
                   [.31591843195509184,.15925708082041834,-.9353249310050448,1.1694769385664476],
                   [0,0,0,1]])
    before=d.qpos.copy()
    controller.step({'base':np.zeros(3),'eef_pose':pose,'gripper':[0.]})
    assert controller.ik_retries>0
    assert controller.max_pose_error<1e-6
    np.testing.assert_array_equal(d.qpos,before)
    limits=controller.joints.robot_view.get_move_group('arm').joint_pos_limits
    assert np.all(controller.seed['arm']>=limits[:,0])
    assert np.all(controller.seed['arm']<=limits[:,1])


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local successful source')
def test_pose_conversion_matches_saved_robot_at_multiple_joint_targets():
    from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController, encode_eef_targets
    from molmo_spaces.robots.robot_views.mobile_franka_droid_view import MobileFrankaDroidRobotView
    m=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'));d=mujoco.MjData(m)
    z=np.load(SOURCE/'initial_state.npz');mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
    mujoco.mj_forward(m,d)
    controller=MolmoSpacesPoseController(m,d)
    with np.load(SOURCE/'actions.npz') as recording:
        commands=recording['ctrl'][::2000].copy()
    poses=encode_eef_targets(controller,commands)
    # Independent FK in the saved scene, not the standalone IK robot.
    probe=mujoco.MjData(m);probe.qpos[:]=d.qpos
    view=MobileFrankaDroidRobotView(probe,'robot_0/')
    for command,pose in zip(commands,poses):
        view.get_move_group('arm').joint_pos=command[controller.groups['arm']]
        mujoco.mj_forward(m,probe)
        np.testing.assert_allclose(pose,view.get_move_group('gripper').leaf_frame_to_robot,atol=1e-9)
