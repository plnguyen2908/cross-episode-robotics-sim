from cross_episode_sim.tasks.composites.execution import recover_pick

FAIL = dict(success=False, planner_rebuilt_after_rollback=True,
            error=dict(message='Grasp unreachable at current dock; call navigate'))
STEP = dict(skill='pick', arguments=dict(target='bowl_01'))

class Interface:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
    def execute_skill(self, skill, arguments):
        self.calls.append((skill, arguments))
        return next(self.replies)

def test_explicit_nav_then_pick():
    interface = Interface([dict(success=True), dict(success=True)])
    result, events = recover_pick(interface, STEP, FAIL, 'test', 1, 16)
    assert result['success']
    assert interface.calls == [('navigate', dict(target='bowl_01', dock='1')),
                               ('pick', dict(target='bowl_01'))]
    assert len(events) == 2

def test_bounded_and_no_infrastructure_retry():
    interface = Interface([dict(success=True), FAIL] * 4)
    result, events = recover_pick(interface, STEP, FAIL, 'test', 1, 16)
    assert not result['success']
    assert len(events) == 8
    broken = dict(success=False, error=dict(message='Broken pipe'))
    interface = Interface([])
    assert recover_pick(interface, STEP, broken, 'test', 1, 16) == (broken, [])


def test_public_pick_stops_before_hidden_tuck_or_navigation():
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pytest
    from cross_episode_sim.manipulation.recovery import RecoveringManipulation
    c = RecoveringManipulation.__new__(RecoveringManipulation)
    c.report = {}
    c.args = SimpleNamespace(annotation_source='droid')
    c._public_skill_navigation = False
    c.base_pose = lambda: __import__('numpy').zeros(3)
    c._plan_with_grasp_fallback = Mock(side_effect=RuntimeError(
        'No annotated grasp passed robot reachability and collision checks'))
    c.tuck_for_navigation = Mock()
    c.navigate = Mock()
    with pytest.raises(RuntimeError, match='Grasp unreachable at current dock'):
        c.select_annotated_grasp()
    c.tuck_for_navigation.assert_not_called()
    c.navigate.assert_not_called()


def test_nearby_recovery_reuses_door_navigation_and_skips_current_pose():
    import numpy as np
    from unittest.mock import Mock, patch
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c = BreakfastEpisode.__new__(BreakfastEpisode)
    c._public_skill_navigation = True
    c._public_dock_skip = 1
    c.base_pose = lambda: np.zeros(3)
    c.dock_candidates = lambda *a, **kw: iter([
        np.array([-.1, 0., 0.]), np.zeros(3), np.array([.1, 0., 0.])])
    c._refresh_room_door_maps = Mock()
    c.plan_route = Mock(return_value='checked')
    c.task_navigate = Mock()
    c.object_name = 'bowl'
    c.object_info = {'bowl': {'role': 'bowl_01'}}
    c.record = Mock()
    with patch('cross_episode_sim.skills.door_motion.dock_if_needed') as reposition:
        c.navigate_to_site('kitchen', np.zeros(3), False)
    assert reposition.call_count == 1
    np.testing.assert_allclose(reposition.call_args.args[1], [.1, 0.])
    assert c.plan_route.call_count == 1
    c.task_navigate.assert_not_called()


def test_spawn_clearance_includes_hinged_appliance_door():
    import mujoco
    import pytest
    from unittest.mock import patch
    from cross_episode_sim.tasks.breakfast.scene import allocate_sites
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="counter_main_main_group_main">
        <geom type="box" pos="0 0 .8" size=".5 .5 .02"/>
      </body>
      <body name="appliance_door" pos="0 0 1.05">
        <joint type="hinge"/><geom type="box" size=".1 .1 .02"/>
      </body>
      <body name="item" pos="2 0 1.5"><freejoint/>
        <geom type="box" size=".05 .05 .05"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    request = dict(body='item', room='kitchen', field='position',
                   info={}, preferred=.5, check_static=True, candidates=[(.5, .025)])
    with patch('cross_episode_sim.tasks.breakfast.scene.site_pose', return_value=[0, 0, .873]):
        allocate_sites(model, data, [request])
        request['pickup_clearance'] = .12
        with pytest.raises(ValueError, match='No separated supported'):
            allocate_sites(model, data, [request])


def test_place_recovery_keeps_held_object_and_destination():
    interface = Interface([dict(success=True), dict(success=True)])
    step = dict(skill='place', arguments=dict(receptacle='blue_basket', region='interior'))
    failure = dict(FAIL, error=dict(message='Placement is unreachable from this dock'))
    previous = [dict(skill='navigate', arguments=dict(target='drink_01', destination='blue_basket'))]
    result, _ = recover_pick(interface, step, failure, 'test', 3, 24, previous)
    assert result['success']
    assert interface.calls[0] == ('navigate', dict(target='drink_01', destination='blue_basket', dock='1'))
    assert interface.calls[1] == ('place', step['arguments'])


def test_knob_recovery_preserves_off_command():
    interface = Interface([dict(success=True), dict(success=True)])
    step = dict(skill='rotate_knob', arguments=dict(target='burner_2_knob', state='off'))
    failure = dict(FAIL, error=dict(message='No nearby placement IK'))
    result, _ = recover_pick(interface, step, failure, 'test', 16, 17)
    assert result['success']
    assert interface.calls[1] == ('rotate_knob', step['arguments'])
