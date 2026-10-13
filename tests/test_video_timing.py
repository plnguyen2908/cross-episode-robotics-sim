"""History/current videos preserve simulated waits and ignore old speedup settings."""
from types import SimpleNamespace

import numpy as np

from cross_episode_sim.controller.reorder_chain import PhysicalReorderCheck
from cross_episode_sim.data.trajectory_cleanup import prepare_video_trace
from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode


def test_breakfast_overrides_legacy_speed_and_keeps_pauses(tmp_path, monkeypatch):
    from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
    def initialize(self, args, selection):
        self.output = tmp_path
        self.report = {'video_speedup': args.video_speedup}
    monkeypatch.setattr(CrossRoomManipulation, '__init__', initialize)
    args = SimpleNamespace(video_speedup=3, lift_height=.12)
    controller = BreakfastEpisode(args, {}, {'coffee': {}})
    assert args.video_speedup == controller.report['video_speedup'] == 1
    assert controller.trim_video_pauses is False
    controller.trace = [{'time': i * .04, 'qpos': [0], 'stage': 'wait'} for i in range(51)]
    assert prepare_video_trace(controller) is controller.trace
    assert not (tmp_path / 'trace_trimmed.json').exists()


def test_deferred_video_records_two_second_stationary_wait(monkeypatch):
    import cross_episode_sim.controller.reorder_chain as module
    monkeypatch.setattr(module.mujoco, 'mj_forward', lambda *args: None)
    frames = []
    controller = SimpleNamespace(
        trace=[{'time': i * .04, 'qpos': [0], 'stage': 'wait'} for i in range(51)],
        trim_video_pauses=False, args=SimpleNamespace(video_speedup=1., video_fps=25),
        data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1)), model=None,
        report={'success': True}, render_video_frame=lambda label, time: frames.append(time))
    PhysicalReorderCheck.render_deferred_video(controller)
    assert len(frames) == 51
    assert frames[-1] == 2.
