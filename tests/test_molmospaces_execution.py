import numpy as np
import mujoco
import pytest
from pathlib import Path
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController

SOURCE=Path('runs/train_test_v1/train/batches/round_0162/clear_and_store_000162/action_recording/model.mjb')


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local saved robot model')
def test_native_absolute_controller_mapping_and_geometry():
    m=mujoco.MjModel.from_binary_path(str(SOURCE));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    originals={f:getattr(m,f).copy() for f in
               ('body_pos','body_mass','body_inertia','geom_size','geom_contype','geom_conaffinity')}
    state=d.qpos.copy()
    backend=MolmoSpacesController(m,d)
    for f,before in originals.items():np.testing.assert_array_equal(getattr(m,f),before)
    for yaw in (-4.,4.):
        action={'base':np.array([1.,2.,yaw]),'arm':np.array([0.,-.5,0.,-2.,0.,1.,0.]),
                'gripper':np.array([100.])}
        backend.step(action)
        for g,controller in backend.controllers.items():
            expected=np.clip(action[g],controller.ctrl_range[:,0],controller.ctrl_range[:,1])
            actual=d.ctrl[backend.groups[g]]
            if g == 'base':
                np.testing.assert_array_equal(actual[:2],expected[:2])
                assert abs(np.arctan2(np.sin(actual[2]-yaw),np.cos(actual[2]-yaw)))<1e-12
            else:
                np.testing.assert_array_equal(actual,expected)
        assert d.ctrl[m.actuator('robot_0/base_theta_act').id]!=0
    np.testing.assert_array_equal(d.qpos,state)  # no teleport or physics step
    previous=d.ctrl.copy()
    with pytest.raises(ValueError):backend.step({**action,'arm':np.zeros(6)})
    np.testing.assert_array_equal(d.ctrl,previous)


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local saved robot model')
def test_native_joint_delta_is_measured_relative_and_gripper_absolute():
    m=mujoco.MjModel.from_binary_path(str(SOURCE));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    backend=MolmoSpacesController(m,d,action_space='joint_delta')
    arm=backend.robot_view.get_move_group('arm')
    # Put the arm at a valid configuration for this unit test only.
    arm.joint_pos=np.array([0.,-.5,0.,-2.,0.,1.,0.])
    mujoco.mj_forward(m,d)
    before=arm.noop_ctrl.copy()
    action={'base':np.zeros(3),'arm':np.full(7,.01),'gripper':np.array([100.])}
    backend.step(action)
    np.testing.assert_allclose(d.ctrl[backend.groups['arm']],before+.01)
    np.testing.assert_allclose(d.ctrl[backend.groups['gripper']],[100.])
    # Same delta must anchor to a changed measured state, not integrate targets.
    arm.joint_pos=before+.02
    mujoco.mj_forward(m,d)
    backend.step(action)
    np.testing.assert_allclose(d.ctrl[backend.groups['arm']],before+.03)


@pytest.mark.skipif(not SOURCE.exists(),reason='Requires local saved robot model')
def test_target_deltas_accumulate_commands_not_tracking_error():
    m=mujoco.MjModel.from_binary_path(str(SOURCE));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    backend=MolmoSpacesController(m,d,action_space='joint_target_delta')
    start=backend.reference['arm'].copy()
    # Native clipping is intentional; choose safe absolute destinations.
    goal=np.array([.1,-.5,.1,-2.,.1,1.,.1])
    first=goal-start
    backend.step({'base':np.zeros(3),'arm':first,'gripper':np.array([100.])})
    np.testing.assert_allclose(d.ctrl[backend.groups['arm']],goal,atol=1e-14)
    # Physics has not moved. Increment is still relative to the prior target.
    backend.step({'base':np.zeros(3),'arm':np.full(7,.01),'gripper':np.array([0.])})
    np.testing.assert_allclose(d.ctrl[backend.groups['arm']],goal+.01,atol=1e-14)


def test_extended_delta_roundtrip_preserves_native_targets():
    from cross_episode_sim.data.target_deltas import encode_target_deltas
    targets=np.array([[1.23456789012345],[.0123],[.75],[.75],[.5]])
    start=np.array([-.23456789012345],dtype=np.longdouble)
    delta=encode_target_deltas(start,targets)
    decoded=np.cumsum(np.vstack((start,delta)),axis=0)[1:].astype(np.float64)
    np.testing.assert_array_equal(decoded,targets)
    # Do not silently qualify a platform/sequence with insufficient precision.
    with pytest.raises(ValueError,match='precision'):
        encode_target_deltas([1.23456789012345],[[1.23e-13]])
