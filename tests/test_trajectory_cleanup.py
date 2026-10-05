"""Editing pauses preserves actions, moving objects, event time and raw states."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cross_episode_sim.data.trajectory_cleanup import trim_stationary_holds, prepare_video_trace


def trace(stage='pregrasp'):
    return [dict(time=i*.04, qpos=[0., 0., 0.], stage=stage,
                 active_object='cup', review_phase='PICK', look_object='cup')
            for i in range(100)]


def test_stationary_hold_removed_but_endpoints_and_original_are_preserved():
    raw = trace(); original = copy.deepcopy(raw)
    rows, report = trim_stationary_holds(raw)
    assert raw == original
    assert rows[0]['source_index'] == 0 and rows[-1]['source_index'] == 99
    assert rows[-1]['time'] == pytest.approx(.04)
    assert report['removed_seconds'] == pytest.approx(3.92)
    assert all(r['qpos'] == raw[r['source_index']]['qpos'] for r in rows)
    assert all(r['source_time'] == raw[r['source_index']]['time'] for r in rows)


@pytest.mark.parametrize('component', [0,1,2])
def test_slow_accumulating_robot_finger_or_object_motion_is_retained(component):
    raw = trace()
    for i,row in enumerate(raw):row['qpos'][component] = i*.00002
    rows, report = trim_stationary_holds(raw)
    assert len(rows) == len(raw) and report['removed_seconds'] == 0.


@pytest.mark.parametrize('stage', ['wait for human filling', 'settle physical food in vessels', 'dynamic change'])
def test_semantic_waits_remain(stage):
    raw = trace(stage)
    rows, report = trim_stationary_holds(raw)
    assert len(rows) == len(raw) and report['removed_seconds'] == 0.


def test_stage_boundaries_and_actual_movement_survive():
    raw = trace()
    for i in range(30,60):
        raw[i]['stage'] = 'lift'
        raw[i]['qpos'][0] = (i-30)*.01
    for i in range(60,100):
        raw[i]['stage'] = 'hold'
        raw[i]['qpos'][0] = .29
    rows, report = trim_stationary_holds(raw)
    kept = {r['source_index'] for r in rows}
    assert set(range(29,61)) <= kept
    assert np.all(np.diff([r['time'] for r in rows]) > 0)
    assert report['removed_seconds'] > 0.


def test_single_sample_and_empty_trace():
    assert trim_stationary_holds([])[0] == []
    rows, summary = trim_stationary_holds(trace()[:1])
    assert len(rows) == 1 and summary['removed_seconds'] == 0.


def test_nonmonotonic_time_rejected():
    raw = trace();raw[-1]['time'] = 0.
    with pytest.raises(ValueError,match='monotonic'):trim_stationary_holds(raw)


def test_output_uses_same_trimmed_rows_without_mutating_raw_trace(tmp_path):
    c = SimpleNamespace(trace=trace(),output=tmp_path,report={},trim_video_pauses=True)
    rows = prepare_video_trace(c)
    assert len(rows) < len(c.trace)
    assert (tmp_path/'trace_trimmed.json').is_file()
    assert c.report['video_trajectory'] == str(tmp_path/'trace_trimmed.json')


def test_replay_schedules_compact_time_but_preserves_source_event_time():
    from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck
    from unittest.mock import patch
    rows = [dict(time=0.,source_time=10.,qpos=[1.],stage='before fill'),
            dict(time=.04,source_time=20.,qpos=[2.],stage='after fill')]
    rendered=[]
    c = SimpleNamespace(trace=rows,args=SimpleNamespace(video_speedup=1.,video_fps=25.),
        data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1),time=0.),model=None,
        report={'success':True}, render_video_frame=lambda label,time:rendered.append(time))
    with patch('cross_episode_sim.controller.reorder_chain.mujoco.mj_forward'):
        PhysicalReorderCheck.render_deferred_video(c)
    assert rendered == [10.,20.]


def test_compact_failure_video_does_not_add_two_second_pause():
    from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck
    from unittest.mock import patch
    rows = [dict(time=0.,qpos=[1.],stage='failed placement')]
    rendered=[]
    c = SimpleNamespace(trace=rows,trim_video_pauses=True,
        args=SimpleNamespace(video_speedup=1.,video_fps=25.),
        data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1),time=0.),model=None,
        report={'success':False}, render_video_frame=lambda label,time:rendered.append(label))
    with patch('cross_episode_sim.data.trajectory_cleanup.prepare_video_trace',return_value=rows), \
            patch('cross_episode_sim.controller.reorder_chain.mujoco.mj_forward'):
        PhysicalReorderCheck.render_deferred_video(c)
    assert len(rendered) == 2  # recorded state, then its failure caption once
    assert rendered[-1].startswith('FAILED:')
