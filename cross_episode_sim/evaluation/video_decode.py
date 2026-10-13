"""Bounded-memory video decoding using the model processor's frame selection."""
from types import MethodType


def stream_video(path, sample_indices_fn):
    import av
    import numpy as np
    import torch
    from transformers.video_utils import VideoMetadata

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count, fps = stream.frames, float(stream.average_rate or 0)
        if count <= 0 or fps <= 0:
            raise ValueError('Native video sampling requires known frame count and frame rate')
        metadata = VideoMetadata(total_num_frames=count, fps=fps, duration=count / fps,
                                 video_backend='pyav_stream', height=stream.height, width=stream.width)
        indices = list(map(int, sample_indices_fn(metadata=metadata)))
        if not indices or min(indices) < 0 or max(indices) >= count:
            raise ValueError('Processor selected invalid video frame indices')
        wanted = set(indices)
        last = max(wanted)
        frames = {}
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format='rgb24')
            if index >= last:
                break
        if set(frames) != wanted:
            raise ValueError('Video ended before the processor-selected frames were decoded')
        video = torch.from_numpy(np.stack([frames[index] for index in indices])).permute(0, 3, 1, 2).contiguous()
        metadata.frames_indices = indices
        return video, metadata


def install_streaming_decoder(processor):
    """Preserve processor sampling/timestamps; replace only its file decoder."""
    def fetch(self, paths, sample_indices_fn=None):
        if isinstance(paths, list):
            return list(zip(*[fetch(self, path, sample_indices_fn) for path in paths]))
        if sample_indices_fn is None:
            raise ValueError('Model processor must supply its sampling function')
        return stream_video(paths, sample_indices_fn)
    processor.video_processor.fetch_videos = MethodType(fetch, processor.video_processor)
