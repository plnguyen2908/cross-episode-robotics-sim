from .rgbd import RGBDMapper, TrackingResult


def create_mapper(backend=None, **kwargs):
    """Select sensor mapping explicitly or with CROSS_SIM_MAPPING_BACKEND.

    The DA3 backend requires the isolated DA3 dependencies and CUDA. It never
    silently falls back to RGB-D tracking, and does not consume truth poses.
    """
    import os
    backend = backend or os.environ.get('CROSS_SIM_MAPPING_BACKEND', 'rgbd')
    if backend == 'rgbd':
        device = os.environ.get('CROSS_SIM_MAPPING_DEVICE', 'CUDA:0')
        if device: kwargs.setdefault('device', device)
        kwargs.setdefault('feature_backend', os.environ.get('CROSS_SIM_RGBD_FEATURES', 'cuda_sift'))
        threads = os.environ.get('CROSS_SIM_RGBD_THREADS', '8')
        if threads:
            import cv2
            if int(threads) < 1: raise ValueError('CROSS_SIM_RGBD_THREADS must be positive')
            cv2.setNumThreads(int(threads))
        from cross_episode_sim.runtime.episode import current_runtime
        runtime=current_runtime()
        if runtime is not None and runtime.artifact_root is not None:
            from .session import TrackingSession
            if runtime.mapper is None:
                kwargs['fuse_dense']=False
                runtime.mapper=TrackingSession(RGBDMapper(**kwargs),global_frame=True)
                runtime.metrics['tracker_initializations']=1
            return runtime.mapper.view()
        return RGBDMapper(**kwargs)
    if backend == 'da3':
        from .da3 import DA3Mapper
        kwargs.setdefault('window', int(os.environ.get('CROSS_SIM_DA3_WINDOW', '16')))
        return DA3Mapper(**kwargs)
    raise ValueError(f'Unknown mapping backend: {backend}')
