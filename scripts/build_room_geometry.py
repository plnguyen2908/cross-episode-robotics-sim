"""Save each visual memory map's history room geometry as merged/<key>/history_geometry.npz.

The geometry-first relocalizer matches a reconstructed scan against this cloud.
It is built from the history episodes' own RGB-D streams and estimated keyframe
poses (reconstruct_history), in the map frame, compacted to 6 cm voxels, so the
published map does not need the raw streams. Never simulator geometry.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from cross_episode_sim.mapping.history_localization import HistoryRelocalizer
from cross_episode_sim.mapping.room_geometry_localization import reconstruct_history

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--streams-root', type=Path, default=Path('/nobackup/le/cross-episode-qwen-capx/runs/visual_memory_balanced_v4'),
                   help='memory root that still has streams/<episode>/sensors.h5')
    p.add_argument('--maps', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory/visual/merged'))
    a = p.parse_args()
    for merged in sorted(a.maps.iterdir()):
        out = merged/'history_geometry.npz'
        if out.exists():
            print('exists', out); continue
        t = time.monotonic()
        source = a.streams_root/'merged'/merged.name
        if json.loads((source/'graph.json').read_text()) != json.loads((merged/'graph.json').read_text()):
            raise SystemExit(f'{merged.name}: graph differs from the stream root copy')
        points = reconstruct_history(HistoryRelocalizer(a.streams_root, source)).astype(np.float32)
        np.savez_compressed(out, points=points, voxel_m=np.float32(.06),
                            episodes=np.array(list(json.loads((merged/'graph.json').read_text())['episodes'])))
        print(merged.name, len(points), f'{time.monotonic()-t:.1f}s', flush=True)
