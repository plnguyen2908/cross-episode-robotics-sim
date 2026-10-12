"""Complete public history videos; sampling belongs to each planner adapter."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def history_context(example):
    """Expose complete video files, never the current demonstration or private metadata."""
    root = Path(example).resolve()
    histories = []
    paths = sorted((p for p in root.glob('history_*.mp4') if p.stem[8:].isdigit()),
                   key=lambda p: int(p.stem[8:]))
    if not paths:
        raise ValueError('No history videos found')
    for path in paths:
        name = path.stem
        if not path.is_file():
            raise FileNotFoundError(path)
        histories.append(dict(name=name, video_path=str(path), mime_type='video/mp4'))
    return dict(instruction=(root / 'prompt.txt').read_text(), histories=histories)


def sample_video(path, frames):
    """Optional adapter utility; the evaluation harness never chooses frames."""
    if type(frames) is not int or not 1 <= frames <= 256:
        raise ValueError('frames must be in [1, 256]')
    reader = imageio.get_reader(path)
    try:
        total = reader.count_frames()
        fps = float(reader.get_meta_data()['fps'])
        if total < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video must contain frames and have a valid frame rate')
        indices = np.unique(np.linspace(0, total-1, min(total, frames), dtype=int))
        return dict(frames=[reader.get_data(int(i)) for i in indices],
                    timestamps_seconds=(indices/fps).tolist())
    finally:
        reader.close()
