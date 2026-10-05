"""Lifecycle tests use fake callbacks, not physical qualification evidence."""
from types import SimpleNamespace
from cross_episode_sim.skills.composite import CompositeEpisode, Operation


def controller(tmp_path):
    events = []
    c = SimpleNamespace(report={}, output=tmp_path, data=SimpleNamespace(time=0.))
    c.finish_run_outputs = lambda success: events.append(('finish', success))
    return c, events


def test_preflight_rejects_entire_plan_before_motion(tmp_path):
    c, events = controller(tmp_path)
    runner = CompositeEpisode(c, {'pick': Operation(lambda: events.append('pick'), lambda: True)}, lambda: {'goal': True})
    assert not runner.run('test', [dict(operation='pick', arguments={}), dict(operation='unknown', arguments={})])
    assert events == [('finish', False)]


def test_goal_is_measured_and_all_steps_precede_render(tmp_path):
    c, events = controller(tmp_path)
    def move():
        events.append('move')
        c.data.time += 1
    runner = CompositeEpisode(c, {'move': Operation(move, lambda: True)}, lambda: {'placed': True, 'closed': True})
    assert runner.run('test', [dict(operation='move', arguments={})] * 2)
    assert events == ['move', 'move', ('finish', True)]
    assert c.data.time == 2


def test_partial_failure_keeps_state_and_renders(tmp_path):
    c, events = controller(tmp_path)
    def failed():
        c.data.time = 4
        raise RuntimeError('dropped')
    runner = CompositeEpisode(c, {'move': Operation(failed, lambda: True)}, lambda: {'goal': True})
    assert not runner.run('test', [dict(operation='move', arguments={})])
    assert c.data.time == 4
    assert events == [('finish', False)]
    assert c.report['error'] == 'dropped'


def test_empty_goal_evidence_cannot_pass(tmp_path):
    c, _ = controller(tmp_path)
    runner = CompositeEpisode(c, {'move': Operation(lambda: None, lambda: True)}, lambda: {})
    assert not runner.run('test', [dict(operation='move', arguments={})])


def test_render_failure_preserves_physics_outcome(tmp_path):
    c, _ = controller(tmp_path)
    def fail_render(success):
        raise RuntimeError('encoder failed')
    c.finish_run_outputs = fail_render
    runner = CompositeEpisode(c, {'move': Operation(lambda: None, lambda: True)}, lambda: {'goal': True})
    assert runner.run('test', [dict(operation='move', arguments={})])
    assert c.report['success'] is True
    assert c.report['video_status'] == 'failed'
    assert (tmp_path / 'composite_result.json').exists()


def test_bad_later_arguments_rejected_before_first_motion(tmp_path):
    c, events = controller(tmp_path)
    runner = CompositeEpisode(c, {'move': Operation(lambda: events.append('move'), lambda: True)}, lambda: {'goal': True})
    assert not runner.run('test', [dict(operation='move', arguments={}), dict(operation='move', arguments={'unexpected': 1})])
    assert events == [('finish', False)]
