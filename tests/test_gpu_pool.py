"""GPU selection and cooperative lease isolation without requiring GPU hardware."""
from cross_episode_sim.generation import gpu_pool


def test_selection_respects_visible_busy_memory_and_load(monkeypatch):
    listing = ('0, GPU-zero, 40000, 0\n1, GPU-one, 50000, 0\n'
               '2, GPU-two, 10000, 0\n3, GPU-three, 6000, 0\n4, GPU-four, 50000, 90\n')
    monkeypatch.setattr(gpu_pool.subprocess, 'check_output',
                        lambda argv, **kw: 'GPU-one\n' if '--query-compute-apps=gpu_uuid' in argv else listing)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-one,2,3,4')
    assert [g['uuid'] for g in gpu_pool.available_gpus()] == ['GPU-two']
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert [g['uuid'] for g in gpu_pool.available_gpus()] == ['GPU-zero', 'GPU-two']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert gpu_pool.available_gpus() == []


def test_leases_do_not_share_gpu_and_release_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_pool.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(gpu_pool, 'available_gpus', lambda: [{'uuid': 'GPU-a'}, {'uuid': 'GPU-b'}])
    with gpu_pool.lease_gpu() as first:
        assert first == 'GPU-a'
        with gpu_pool.lease_gpu() as second:
            assert second == 'GPU-b'
    try:
        with gpu_pool.lease_gpu() as selected:
            assert selected == 'GPU-a'
            raise ValueError('worker failed')
    except ValueError:
        pass
    with gpu_pool.lease_gpu() as selected:
        assert selected == 'GPU-a'
