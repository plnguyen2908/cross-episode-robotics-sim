"""Empty tuck never executes a waypoint before both legs pass mesh checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.controller.navigation import NavigationTransfer


def controller(reject_all=False):
    c = NavigationTransfer.__new__(NavigationTransfer)
    c.profile = SimpleNamespace(arm_joints=['wrist'], arm_actuators=['act'],
                                torso_joints=[], torso_actuators=[])
    c.embodiment = SimpleNamespace(travel_posture=lambda: [0.])
    c.args = SimpleNamespace(kitchen=True, motion_slowdown=1.)
    c.model = SimpleNamespace(opt=SimpleNamespace(timestep=.01),
                              joint=lambda n: SimpleNamespace(range=np.array([-1., 2.])))
    c.data = SimpleNamespace(joint=lambda n: SimpleNamespace(qpos=[1.]), ctrl=np.array([1.]))
    c.executed = []
    c.tick = lambda dt: c.executed.append(c.data.ctrl.copy())
    c.record = lambda **kw: None
    c.angle = lambda: 0.
    c.actuator_ids = lambda names: [0]
    c.load_world = lambda: None
    c.make_planner = lambda: SimpleNamespace(names=['wrist'], dt=.01,
        plan_joints=lambda start, goal: np.array([start, goal]))
    c.checked = []
    def check(path):
        c.checked.append(path.copy())
        assert not c.executed
        if reject_all or len(path) == 2:
            raise RuntimeError('Tuck path intersects actual geometry')
    c.preflight_empty_arm_trajectory = check
    return c


def test_tuck_checks_both_legs_before_execution():
    c = controller()
    c.tuck_arm()
    assert len(c.checked[-1]) == 4
    assert len(c.executed) == 5
    np.testing.assert_allclose(c.executed[-1], [0.])


def test_rejected_waypoints_do_not_move_robot():
    c = controller(reject_all=True)
    with pytest.raises(RuntimeError, match='No actual-mesh-clear tuck path'):
        c.tuck_arm()
    assert not c.executed
    np.testing.assert_allclose(c.data.ctrl, [1.])
