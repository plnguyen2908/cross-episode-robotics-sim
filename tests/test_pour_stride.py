from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cross_episode_sim.tasks.coffee.pour import MoonlakeRobotPour


@pytest.mark.parametrize('stride,stops', [(5, 21), (10, 11), (15, 7), (20, 6), (30, 4)])
def test_faster_pour_keeps_full_screening_and_final_tilt(stride, stops):
    c = MoonlakeRobotPour.__new__(MoonlakeRobotPour)
    c.spec = dict(pour_angles_deg=list(range(0, 106, 5)), pour_yaws_deg=[0.],
                  pour_execute_stride_deg=stride, withdrawal_execute_stride_deg=15)
    c.grasp_relative = np.eye(4)
    c.profile = SimpleNamespace(namespace='robot/')
    c.data = SimpleNamespace(joint=lambda name: SimpleNamespace(qpos=[0.]))
    c.planner = SimpleNamespace(names=['j1'], plan=Mock(return_value=[[0.]]))
    c.embodiment = SimpleNamespace(planner_tool_offset=lambda: 0.)
    c.pour_pose = lambda *args: np.eye(4)
    c.nearby_ik = Mock(return_value=[0.])
    c.check_loaded_tuck_path = Mock()
    c.move = Mock()
    c.mesh_contact_move = Mock()
    c.tick = Mock()
    c.record = Mock()
    c.report = {}
    c.preplanned_moves = {}
    c.tcp = c.bread_pose = lambda: np.eye(4)
    c.inventory = lambda: dict(hopper=list(range(32)), vessel=[], spilled=[])
    c.grounds_ready = lambda: True
    c._pour_grounds_once()
    assert len(c.check_loaded_tuck_path.call_args_list[0].args[0]) == 22
    angles = c.report['pour_execution_angles_deg']
    assert len(angles) == stops
    assert angles[-1] == 105
    assert c.report['pour_evidence']['counts']['spilled'] == 0
    assert c.poured and not c.pouring


def test_launcher_applies_pour_stride_to_both_coffee_episodes_only():
    from cross_episode_sim.tasks.full_breakfast.__main__ import command, sample
    _, settings = sample(8)
    for task in ('coffee', 'full_breakfast'):
        argv = command(task, settings[task], 'unused', 'test', pour_step_deg=10)
        assert argv[argv.index('--pour-step-deg')+1] == '10'
    assert '--pour-step-deg' not in command('breakfast', settings['breakfast'], 'unused', 'test', 10)
