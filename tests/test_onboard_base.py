from types import SimpleNamespace
import numpy as np
from cross_episode_sim.control import onboard_base


def test_base_handoff_preserves_command_not_binary_aperture_guess(monkeypatch):
    class Gripper:
        ctrl=np.array([255.])
        @property
        def noop_ctrl(self):
            raise AssertionError('A wide grasp must not be interpreted as open')
    groups={'gripper':Gripper(),'base':SimpleNamespace(noop_ctrl=np.zeros(3)),
            'arm':SimpleNamespace(noop_ctrl=np.zeros(7))}
    controller=SimpleNamespace(robot_view=SimpleNamespace(get_move_group=groups.__getitem__))
    monkeypatch.setattr(onboard_base,'MolmoSpacesController',lambda *args:controller)
    model=SimpleNamespace(joint=lambda name:SimpleNamespace(qposadr=[{'base_x':0,'base_y':1,'base_theta':2}[name.split('/')[-1]]]))
    base=onboard_base.OnboardBaseActuator(model,SimpleNamespace(qpos=np.zeros(3)))
    np.testing.assert_array_equal(base.hold['gripper'],[255.])
    groups['gripper'].ctrl[:]=0
    np.testing.assert_array_equal(base.hold['gripper'],[255.])


def test_lateral_commands_need_planar_guard_and_use_body_frame(monkeypatch):
    import pytest
    commands=[]
    groups={g:SimpleNamespace(noop_ctrl=np.zeros(3 if g=='base' else 7),ctrl=np.array([255.]),joint_pos=np.zeros(7)) for g in ('base','arm','gripper')}
    controller=SimpleNamespace(robot_view=SimpleNamespace(get_move_group=groups.__getitem__),step=lambda a:commands.append(a))
    monkeypatch.setattr(onboard_base,'MolmoSpacesController',lambda *args:controller)
    monkeypatch.setattr(onboard_base.mujoco,'mj_step',lambda *args:None)
    model=SimpleNamespace(opt=SimpleNamespace(timestep=.002),joint=lambda name:SimpleNamespace(qposadr=[{'base_x':0,'base_y':1,'base_theta':2}[name.split('/')[-1]]]))
    base=onboard_base.OnboardBaseActuator(model,SimpleNamespace(qpos=np.array([0.,0.,np.pi/2])),motion_guard=lambda *args:True)
    with pytest.raises(RuntimeError,match='planar sweep guard'):base.step(0,0,lateral=.1)
    checks=[]
    base.step(0,0,lateral=.1,planar_motion_guard=lambda *args:checks.append(args) or True)
    assert checks==[(0,.1,0,.02)]
    np.testing.assert_allclose(commands[-1]['base'],[-.002,0,np.pi/2],atol=1e-12)
    with pytest.raises(RuntimeError,match='rejected'):base.step(0,0,lateral=.1,planar_motion_guard=lambda *args:False)

    base.step(0,0)
    base.step(.1,0)
    np.testing.assert_allclose(commands[-1]['base'],[0,.002,np.pi/2],atol=1e-12)

    # Planned pauses retain the validated target so servo lag can settle.
    target=commands[-1]['base'].copy()
    base.step(0,0,retain_target=True)
    base.step(0,0,retain_target=True)
    np.testing.assert_allclose(commands[-1]['base'],target)
    # An ordinary stop still cancels the pending target.
    base.step(0,0)
    np.testing.assert_allclose(commands[-1]['base'],[0,0,np.pi/2])
