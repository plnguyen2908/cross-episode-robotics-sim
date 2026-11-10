import cv2
import cross_episode_sim.slam as slam


def test_pipeline_defaults_and_explicit_overrides(monkeypatch):
    for key in ('CROSS_SIM_MAPPING_DEVICE','CROSS_SIM_RGBD_FEATURES','CROSS_SIM_RGBD_THREADS','CROSS_SIM_MAPPING_BACKEND'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(slam,'RGBDMapper',lambda **kwargs:kwargs)
    monkeypatch.setattr(cv2,'setNumThreads',lambda n:threads.append(n))
    threads=[]
    assert slam.create_mapper()=={'device':'CUDA:0','feature_backend':'cuda_sift'}
    assert threads==[8]
    assert slam.create_mapper(device='CPU:0',feature_backend='opencv')['device']=='CPU:0'
    monkeypatch.setenv('CROSS_SIM_MAPPING_DEVICE','CPU:0')
    monkeypatch.setenv('CROSS_SIM_RGBD_FEATURES','opencv')
    monkeypatch.setenv('CROSS_SIM_RGBD_THREADS','4')
    assert slam.create_mapper()=={'device':'CPU:0','feature_backend':'opencv'}
    assert threads[-1]==4
