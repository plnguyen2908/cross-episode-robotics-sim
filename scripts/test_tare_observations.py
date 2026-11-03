"""Feed cached onboard observations + visual estimates to upstream TARE.

This tests the planner interface only. It does NOT execute the proposed path,
and its report must not be described as a physical exploration success.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from cross_episode_sim.navigation.tare import TarePlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--estimates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.stream/'calibration.json').read_text())
    depth = np.load(args.stream/'depth.npy', mmap_mode='r')
    masks = np.load(args.stream/'robot_mask.npy', mmap_mode='r')
    times = np.load(args.stream/'timestamps.npy')
    poses = np.load(args.estimates/'all_frame_estimates.npy')
    records = json.loads((args.estimates/'tracking.json').read_text())
    if len(poses) != len(times) or not np.allclose(times, [r['timestamp'] for r in records]):
        raise ValueError('Visual estimates must match the observation stream timestamps')
    K = np.array(meta['K']); mount = np.array(meta['T_base_optical'])
    outputs = []
    observed = []
    with TarePlanner(args.output/'tare') as planner:
        indices = np.flatnonzero(times-times[0] <= args.seconds)[::2]
        for i in tqdm(indices, desc='Upstream TARE sensor replay'):
            if not records[i]['accepted']:
                continue
            camera = SimpleNamespace(depth_m=depth[i], valid_depth=np.isfinite(depth[i]),
                                     K=K, T_base_optical=mount, robot_mask=masks[i])
            tracking = SimpleNamespace(timestamp=float(times[i]), accepted=True, T_map_base=poses[i])
            outputs.append(planner.update(camera, tracking))
            if i % 20 == 0:
                from cross_episode_sim.navigation.tare import observation_payload
                observed.extend(observation_payload(camera, tracking, stride=12)['points'])
    (args.output/'responses.json').write_text(json.dumps(outputs, indent=2))
    report = dict(kind='planner_interface_only_not_physical_execution', observations=len(outputs),
                  path_updates=max((x['sequence'] for x in outputs), default=0),
                  nonempty_paths=sum(bool(x['path']) for x in outputs),
                  upstream_reported_finished=any(x['finished'] for x in outputs),
                  physical_success=None)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    points = np.asarray(observed)
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=.2, cmap='Greys', alpha=.4)
    paths = [x for x in outputs if len(x['path']) > 1]
    if paths:
        path = np.asarray(paths[-1]['path'])
        ax.plot(path[:, 0], path[:, 1], 'r.-', label='Latest upstream proposed path (not executed)')
    valid = np.isfinite(poses).all(axis=(1, 2)) & (times-times[0] <= args.seconds)
    ax.plot(poses[valid, 0, 3], poses[valid, 1, 3], color='blue', label='Visual-SLAM input trajectory')
    ax.set_aspect('equal'); ax.legend(); ax.set_title('TARE interface test: observed RGB-D only')
    ax.set_xlabel('SLAM x (m)'); ax.set_ylabel('SLAM y (m)')
    fig.savefig(args.output/'proposed_path.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
