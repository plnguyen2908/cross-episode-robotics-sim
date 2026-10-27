"""Native pose-controller checks using a saved scene, without changing physics."""
from pathlib import Path
import mujoco
import numpy as np
import pytest

SOURCE=Path('runs/train_test_v1/train/batches/round_0162/set_table_000162/action_recording')


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local successful source')
def test_native_pose_frame_and_no_state_reset():
    from cross_episode_sim.finetuning.molmospaces_eef import MolmoSpacesPoseController
    m=mujoco.MjModel.from_binary_path(str(SOURCE/'model.mjb'));d=mujoco.MjData(m)
    z=np.load(SOURCE/'initial_state.npz');mujoco.mj_setState(m,d,z['state'],int(z['state_type']))
    mujoco.mj_forward(m,d)
    controller=MolmoSpacesPoseController(m,d)
    view=controller.joints.robot_view
    pose=controller.kinematics.fk(controller.seed,np.eye(4),rel_to_base=True)['gripper'].copy()
    physical=view.get_move_group('gripper').leaf_frame_to_robot
    np.testing.assert_allclose(pose,physical,atol=1e-9)
    before=d.qpos.copy();time=d.time
    controller.step({'base':view.get_move_group('base').noop_ctrl,'eef_pose':pose,
                     'gripper':view.get_move_group('gripper').noop_ctrl})
    np.testing.assert_array_equal(d.qpos,before)
    assert d.time==time
    assert controller.max_pose_error<1e-4
    with pytest.raises(ValueError,match='rigid pose'):
        bad=pose.copy();bad[0,0]*=2
        controller.step({'base':np.zeros(3),'eef_pose':bad,'gripper':[0.]})
