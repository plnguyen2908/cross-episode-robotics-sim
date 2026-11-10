import pytest
from cross_episode_sim.runtime.worker_endpoints import worker_endpoints


def test_shared_temporal_worker_is_rejected_even_if_rpcs_are_serial():
    with pytest.raises(ValueError,match='distinct One-Ring'):
        worker_endpoints(['0','5'],{'onering':'localhost:1','molmo':'localhost:2'})


def test_workers_share_stateless_models_but_have_independent_navigation():
    shared={'onering':'localhost:1','molmo':'localhost:2'}
    result=worker_endpoints(['0','5'],shared,{'5':{'onering':'localhost:3'}})
    assert result['0']['onering']=='localhost:1'
    assert result['5']['onering']=='localhost:3'
    assert result['0']['molmo']==result['5']['molmo']=='localhost:2'
    assert shared['onering']=='localhost:1'


def test_single_worker_preserves_existing_configuration():
    assert worker_endpoints(['0'],{'onering':'one'})=={'0':{'onering':'one'}}


@pytest.mark.parametrize('gpus,overrides', [(['0','0'],{}), (['0'],{'5':{}})])
def test_invalid_slots(gpus,overrides):
    with pytest.raises(ValueError):worker_endpoints(gpus,{'onering':'one'},overrides)
