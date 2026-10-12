import pytest

from cross_episode_sim.rendering_gpu import match_devices, worker_environment


def test_reordered_egl_devices_match_physical_pci_addresses():
    nvidia = [dict(index='0', uuid='GPU-first', pci='00000000:01:00.0'),
              dict(index='4', uuid='GPU-second', pci='00000000:A1:00.0')]
    egl = [dict(index=0, pci='0000:a1:00.0'), dict(index=3, pci='0000:01:00.0')]
    bindings = match_devices(['0', '4'], nvidia, egl)
    assert bindings['0'] == dict(cuda_uuid='GPU-first', egl_device=3)
    assert bindings['4'] == dict(cuda_uuid='GPU-second', egl_device=0)
    env = dict(CUDA_VISIBLE_DEVICES='7', MUJOCO_EGL_DEVICE_ID='2', MUJOCO_GL='egl')
    worker = worker_environment(env, '4', bindings['4'])
    assert worker['CUDA_VISIBLE_DEVICES'] == 'GPU-second'
    assert worker['MUJOCO_EGL_DEVICE_ID'] == '0'
    assert env['CUDA_VISIBLE_DEVICES'] == '7'
    assert env['MUJOCO_EGL_DEVICE_ID'] == '2'


def test_missing_renderer_fails_before_starting_workers():
    nvidia = [dict(index='0', uuid='GPU-first', pci='00000000:01:00.0')]
    with pytest.raises(RuntimeError, match='No EGL rendering device matches GPU 0'):
        match_devices(['0'], nvidia, [])
    with pytest.raises(ValueError, match='does not identify one'):
        match_devices(['7'], nvidia, [])


def test_uuid_selection_and_non_egl_backend():
    nvidia = [dict(index='0', uuid='GPU-first', pci='00000000:01:00.0')]
    assert match_devices(['GPU-fir'], nvidia, [dict(index=3, pci='0000:01:00.0')])[
        'GPU-fir']['cuda_uuid'] == 'GPU-first'
    assert worker_environment({'MUJOCO_GL': 'osmesa'}, '4') == {
        'MUJOCO_GL': 'osmesa', 'CUDA_VISIBLE_DEVICES': '4'}
