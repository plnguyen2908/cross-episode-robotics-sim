"""Run upstream VGGT-SLAM on saved onboard RGB; no simulator truth is loaded.

Depth is used only to report a first-frame metric scale, not by upstream SLAM.
Score the output separately, after this process exits.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--submap-size', type=int, default=8)
    p.add_argument('--dropout', type=int, nargs=2)
    p.add_argument('--mask-robot', action='store_true')
    p.add_argument('--min-disparity', type=float, default=50.)
    p.add_argument('--max-keyframe-gap', type=int, default=0)
    a = p.parse_args()
    for path in [a.upstream, a.upstream/'third_party/vggt', a.upstream/'third_party/salad']:
        sys.path.insert(0, str(path.resolve()))
    from vggt_slam.solver import Solver
    from vggt.models.vggt import VGGT
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=False)
    imgs = a.output/'images'
    imgs.mkdir()
    rgb = np.load(a.stream/'rgb.npy', mmap_mode='r')
    mask = np.load(a.stream/'robot_mask.npy', mmap_mode='r')
    solver = Solver(init_conf_threshold=25., lc_thres=.95)
    model = VGGT()
    model.load_state_dict(torch.load(a.weights, map_location='cpu'))
    model = model.eval().to(dtype=torch.bfloat16, device='cuda')
    pending, batches = [], []
    start = time.monotonic()
    first_depth = None
    def process():
        nonlocal first_depth
        t = time.monotonic()
        pred = solver.run_predictions(pending, model, 1, None, None)
        if first_depth is None:
            first_depth = pred['depth'][0].squeeze().copy()
            np.save(a.output/'first_predicted_depth.npy', first_depth)
            np.save(a.output/'first_predicted_K.npy', pred['intrinsic'][0])
        solver.add_points(pred)
        solver.graph.optimize()
        solver.map.write_poses_to_file(str(a.output/'poses.txt'), solver.graph)
        (a.output/f'poses_after_{len(batches):03d}.txt').write_text((a.output/'poses.txt').read_text())
        batches.append({'frames': [int(Path(x).stem) for x in pending],
                        'wall_seconds': time.monotonic()-t,
                        'loop_count': solver.graph.get_num_loops()})
        (a.output/'batches.json').write_text(json.dumps(batches, indent=2))
    last_selected = -1
    for i in tqdm(range(len(rgb)), desc='VGGT onboard frames'):
        if a.dropout and a.dropout[0] <= i < a.dropout[1]:
            continue
        frame = rgb[i].copy()
        if a.mask_robot:
            frame[mask[i]] = 0
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        selected = solver.flow_tracker.compute_disparity(bgr, a.min_disparity, False)
        if a.max_keyframe_gap and i-last_selected >= a.max_keyframe_gap:
            selected = True
            solver.flow_tracker.initialize_keyframe(bgr)
        if selected:
            path = imgs/f'{i:06d}.png'
            cv2.imwrite(str(path), bgr)
            pending.append(str(path))
            last_selected = i
        if len(pending) >= a.submap_size+1:
            process()
            pending = pending[-1:]
    if len(pending) > 1:
        process()
    # First-frame scale uses ONLY the observed depth, never a reference path.
    depth = np.load(a.stream/'depth.npy', mmap_mode='r')[0].copy()
    depth[mask[0]] = np.nan
    h, w = first_depth.shape
    # Upstream crop mode: resize width to 518, height to nearest patch multiple.
    measured = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(measured) & (measured > .65) & (measured < 6.) & (first_depth > 0)
    scale = float(np.median(measured[valid]/first_depth[valid]))
    solver.map.write_points_to_file(solver.graph, str(a.output/'map.ply'))
    report = {'frames': len(rgb), 'submaps': len(batches), 'batches': batches,
              'wall_seconds': time.monotonic()-start,
              'first_frame_depth_scale': scale,
              'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9,
              'dropout': a.dropout, 'mask_robot': a.mask_robot,
              'min_disparity': a.min_disparity, 'max_keyframe_gap': a.max_keyframe_gap,
              'note': 'Upstream RGB-only SLAM. Depth scale is postprocessing from first observed frame. No simulator truth.'}
    (a.output/'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
