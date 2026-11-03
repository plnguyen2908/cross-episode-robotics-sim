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
