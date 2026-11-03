"""Offline diagnostic scorer: simulator truth never enters the SLAM process."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--diagnostics', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run/'run.json').read_text())
    rows = np.loadtxt(a.run/'poses.txt')
    # Overlap frames are duplicated by upstream export; retain last estimate.
    rows = np.array(list({int(r[0]): r for r in rows}.values()))
    rows = rows[np.argsort(rows[:, 0])]
    ids = rows[:, 0].astype(int)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4]
    relative = np.linalg.inv(poses[0]) @ poses
    relative[:, :3, 3] *= info['first_frame_depth_scale']
    mount = np.array(json.loads((a.stream/'calibration.json').read_text())['T_base_optical'])
    est = mount @ relative @ np.linalg.inv(mount)
    all_truth = np.load(a.diagnostics/'base_poses.npy')
    truth = all_truth[ids]
    baseline = np.load(a.baseline/'all_frame_estimates.npy')[ids]
    def metrics(x, reference=truth):
        error = np.linalg.norm(x[:, :3, 3]-reference[:, :3, 3], axis=1)
        angle = np.arctan2(x[:, 1, 0], x[:, 0, 0])-np.arctan2(reference[:, 1, 0], reference[:, 0, 0])
        angle = np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle)))
        return dict(position_rmse_m=float(np.sqrt(np.mean(error**2))),
                    last_keyframe_error_m=float(error[-1]),
                    yaw_rmse_deg=float(np.sqrt(np.mean(angle**2))))
    # Optional benchmark-only similarity alignment; never a deployable scale.
    x, y = est[:, :3, 3], truth[:, :3, 3]
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, s, vt = np.linalg.svd(yc.T@xc/len(x))
    d = np.eye(3); d[2, 2] = np.linalg.det(u@vt)
    rot = u@d@vt
    scale = np.sum(s*np.diag(d))/np.mean(np.sum(xc*xc, axis=1))
    aligned = scale*(xc@rot.T)+y.mean(0)
    report = {'keyframes': len(ids), 'last_keyframe_index': int(ids[-1]),
              'vggt_first_observed_depth_scaled': metrics(est),
              'baseline_no_dropout_same_keyframes': metrics(baseline),
              'diagnostic_only_sim3_aligned_rmse_m': float(np.sqrt(np.mean(np.sum((aligned-y)**2, axis=1)))),
              'note': 'VGGT final optimized keyframes, not continuous online poses. Similarity alignment uses truth ONLY for diagnostic scoring.'}
    causal_poses, causal_ids = [], []
    for snapshot in sorted(a.run.glob('poses_after_*.txt')):
        row = np.atleast_2d(np.loadtxt(snapshot))[-1]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        pose = np.linalg.inv(poses[0])@pose
        pose[:3, 3] *= info['first_frame_depth_scale']
        causal_poses.append(mount@pose@np.linalg.inv(mount))
        causal_ids.append(int(row[0]))
    if causal_poses:
        causal_poses = np.array(causal_poses)
        report['causal_latest_keyframe_per_submap'] = metrics(causal_poses, all_truth[causal_ids])
        report['causal_samples'] = [dict(index=i, position_error_m=float(np.linalg.norm(q[:3, 3]-all_truth[i, :3, 3]))) for i, q in zip(causal_ids, causal_poses)]
    np.save(a.run/'estimated_base_keyframes.npy', est)
    np.save(a.run/'keyframe_ids.npy', ids)
    (a.run/'comparison.json').write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(y[:, 0], y[:, 1], 'k-', label='Reference (scoring only)')
    axes[0].plot(baseline[:, 0, 3], baseline[:, 1, 3], color='tab:blue', label='RGB-D baseline (no dropout)')
    axes[0].plot(x[:, 0], x[:, 1], color='tab:orange', label='VGGT + first-depth scale')
    axes[0].set_aspect('equal'); axes[0].legend(); axes[0].set_title('Estimated base path (metres)')
    axes[1].plot(ids/20, np.linalg.norm(x-y, axis=1), color='tab:orange', label='VGGT')
    axes[1].plot(ids/20, np.linalg.norm(baseline[:, :3, 3]-y, axis=1), color='tab:blue', label='RGB-D baseline (no dropout)')
    if info['dropout']:
        axes[1].axvspan(info['dropout'][0]/20, info['dropout'][1]/20, color='gray', alpha=.3, label='VGGT missing frames')
    axes[1].set(xlabel='Recorded time (s)', ylabel='Position error (m)'); axes[1].legend()
    fig.tight_layout(); fig.savefig(a.run/'comparison.png', dpi=160)
    import open3d as o3d
    fig = plt.figure(figsize=(12, 6))
    clouds = []
    for file, title in [(a.baseline/'map.ply', 'RGB-D baseline'), (a.run/'map.ply', 'VGGT-SLAM 2.0 + depth scale')]:
        cloud = o3d.io.read_point_cloud(str(file))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if file.parent == a.run:
            anchor = np.linalg.inv(poses[0])
            points = points@anchor[:3, :3].T+anchor[:3, 3]
            points *= info['first_frame_depth_scale']
            points = points@mount[:3, :3].T+mount[:3, 3]
        finite = np.isfinite(points).all(axis=1)
        points, colors = points[finite], colors[finite]
        stride = max(1, len(points)//50000)
        clouds.append((points[::stride], colors[::stride], title))
    for n, (points, colors, title) in enumerate(clouds):
        ax = fig.add_subplot(1, 2, n+1, projection='3d')
        ax.scatter(*points.T, c=colors, s=.4, rasterized=True)
        ax.view_init(elev=55, azim=-65)
        ax.set(title=title, xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        bounds = np.quantile(points, [.01, .99], axis=0)
        ax.set_xlim(bounds[:, 0]); ax.set_ylim(bounds[:, 1]); ax.set_zlim(bounds[:, 2])
        ax.set_box_aspect(np.maximum(bounds[1]-bounds[0], .1))
    fig.suptitle('Observed reconstruction — independent axes; outer 1% cropped for display')
    fig.tight_layout(); fig.savefig(a.run/'reconstruction.png', dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
