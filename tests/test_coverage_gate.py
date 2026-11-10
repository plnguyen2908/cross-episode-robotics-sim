from types import SimpleNamespace
import pytest
from cross_episode_sim.navigation.coverage import exploration_request, UNKNOWN_SPACE

@pytest.mark.parametrize('reason', sorted(UNKNOWN_SPACE))
def test_unknown_space_requests_separate_exploration(reason):
    grid=SimpleNamespace(fixed_obstacle_points=[])
    result=exploration_request(grid,dict(reason=reason),'one_ring')
    assert result['status']=='exploration_required'
    assert result['forward']==result['yaw']==0
    assert result['exploration_request']['resume_skill']=='one_ring'

@pytest.mark.parametrize('reason',['observed_3d_collision','depth_braking_arc','tracking_lost','reachability_unverified'])
def test_other_failures_do_not_request_exploration(reason):
    assert exploration_request(SimpleNamespace(fixed_obstacle_points=[]),dict(reason=reason),'docking') is None

def test_mapping_mode_not_interrupted():
    assert exploration_request(SimpleNamespace(),dict(reason='unobserved_base_support'),'docking') is None
