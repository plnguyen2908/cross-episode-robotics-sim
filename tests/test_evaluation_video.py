from pathlib import Path
from types import SimpleNamespace

import pytest

from cross_episode_sim.evaluation.sessions import ControllerSession


@pytest.mark.parametrize('success', [False, True])
def test_rollout_finalization_uses_evaluated_outcome_and_closes_once(tmp_path, success):
    closed = []
    class Resource:
        def close(self):
            closed.append(self)
    resources = [Resource() for _ in range(3)]
    c = SimpleNamespace(output=tmp_path, report={}, args=SimpleNamespace(video_speedup=3),
                        trim_video_pauses=True, video_filename='source.mp4',
                        writer=resources[0], head_writer=resources[1], renderer=resources[2])
    def finish(completed):
        assert completed is success
        assert c.report['composite_goal_evidence'] == {'served': success}
        assert c.args.video_speedup == 1 and c.trim_video_pauses is False
        (tmp_path/'source.mp4').write_bytes(b'recorded rollout')
        c.report['video_status'] = 'complete'
        for resource in resources:
            resource.close()
    c.finish_run_outputs = finish
    session = ControllerSession.__new__(ControllerSession)
    session._c = c; session._renderer = None
    result = session.finalize_video(dict(success=success, goals={'served': success},
                                         termination='model_returned'), tmp_path/'episode')
    assert result['status'] == 'complete'
    assert (tmp_path/'episode'/result['path']).read_bytes() == b'recorded rollout'
    session.close()
    assert closed == resources


def test_failed_render_can_still_close_session(tmp_path):
    def finish(completed):
        raise RuntimeError('Encoder failed')
    c = SimpleNamespace(output=Path(tmp_path), report={}, args=SimpleNamespace(), finish_run_outputs=finish)
    session = ControllerSession.__new__(ControllerSession)
    session._c = c; session._renderer = None
    with pytest.raises(RuntimeError, match='Encoder failed'):
        session.finalize_video(dict(success=False), tmp_path)
    session.close()
