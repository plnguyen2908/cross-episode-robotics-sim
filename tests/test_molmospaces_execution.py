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
