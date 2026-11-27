"""One in-place 360-degree scan from a saved test start, then visual-memory relocalization.

The robot turns with the same LocalScanSkill exploration starts with; the
collector sees only onboard RGB-D and SLAM poses. Afterwards, and only for
scoring, the true start pose is compared with the memory frame origin (the
first history episode's start pose).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
import argparse, json, time
from pathlib import Path

import cv2, mujoco, numpy as np

from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
from cross_episode_sim.evaluation.text_memory import house_of
from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
from cross_episode_sim.sensors.onboard import OnboardSensorInterface
from cross_episode_sim.slam import create_mapper


def base_pose(model, data):
    body = model.body('robot_0/base').id
    T = np.eye(4); T[:3, :3] = data.xmat[body].reshape(3, 3); T[:3, 3] = data.xpos[body]
    return T


PLAN = (Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/history_streams'/'plan.json')


def reference_origin(memory_root, merged):
    """Private scoring only: world pose of the memory frame's origin."""
    graph = json.loads((merged/'graph.json').read_text()); reference = next(iter(graph['episodes']))
    if reference.startswith('tour_'):  # planned-tour map: its frame is the tour's start pose
        tours = (Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/visual_tours')
        house = reference[len('tour_'):].split('_loop')[0]  # loops of one house share its start pose
        return np.load(tours/house/'private_start_pose.npy'), reference
    plan = json.loads(((memory_root/'plan.json') if (memory_root/'plan.json').is_file() else PLAN).read_text())
    source = Path(next(e['source'] for e in plan['episodes'] if e['id'] == reference))/'action_recording'
    model = mujoco.MjModel.from_binary_path(str(source/'model.mjb')); data = mujoco.MjData(model)
    with np.load(source/'actions.npz') as a: data.qpos[:] = a['qpos'][0]
    mujoco.mj_kinematics(model, data)
    return base_pose(model, data), reference


def yaw(T): return np.arctan2(T[1, 0], T[0, 0])


def figure(output, reloc, expected, estimate, audit):
    """Private diagnostic figure: one scan view, the rebuilt room, and both alignments."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    q = reloc.query; h = reloc.history_points
    h2 = h[(h[:, 2] > .3) & (h[:, 2] < 2.)][::3]
    def place(T): return q@T[:3, :3].T+T[:3, 3]
    fig = plt.figure(figsize=(16, 13))
    ax = fig.add_subplot(2, 2, 1); ax.imshow(reloc.cameras[len(reloc.cameras)//2].rgb); ax.axis('off')
    ax.set_title(f'1. Rotation scan: {len(reloc.cameras)} onboard RGB-D views (one shown)')
    ax = fig.add_subplot(2, 2, 2, projection='3d'); sub = q[::max(1, len(q)//6000)]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], s=1, cmap='viridis'); ax.set_title('2. Reconstructed current room (scan-start frame)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    for k, (T, title) in enumerate(((estimate, '3. Matched to history map (estimate)'),
                                    (expected, '4. Check: rebuilt room placed at the TRUE pose (scoring only)'))):
        ax = fig.add_subplot(2, 2, 3+k); ax.scatter(h2[:, 0], h2[:, 1], s=.3, c='0.6', label='history geometry')
        if T is not None:
            p = place(T); ax.scatter(p[::2, 0], p[::2, 1], s=.6, c='tab:orange', label='current reconstruction')
            ax.plot(T[0, 3], T[1, 3], 'bx', ms=12, mew=3, label='robot')
        else: ax.text(.5, .5, 'not localized', transform=ax.transAxes, ha='center', fontsize=16)
        ax.plot(expected[0, 3], expected[1, 3], 'g+', ms=14, mew=2, label='true robot position')
        ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=8); ax.set_title(title); ax.set_xlabel('map x (m)'); ax.set_ylabel('map y (m)')
    fig.suptitle(f"{audit['house']} {audit['task']} — {audit['status']}; error "
                 + (f"{audit['position_error_m']*100:.1f} cm / {audit['yaw_error_degrees']:.2f} deg" if 'position_error_m' in audit else 'n/a')
                 + f"; at true pose {audit['truth_fraction_within_15cm']*100:.0f}% of rebuilt points within 15 cm of history", fontsize=13)
    fig.tight_layout(); fig.savefig(output/'relocalization.png', dpi=80); plt.close(fig)


def run(source, memory_root, output):
    output.mkdir(parents=True, exist_ok=False)
    house, _ = house_of(source); task = json.loads((source/'config.json').read_text())['task']
    merged = memory_directory(memory_root, house, task)
    model, data, _ = load_saved_scene(source)
    truth_start = base_pose(model, data)   # private; never passed to the policy
    base = OnboardBaseActuator(model, data); slam = create_mapper(fuse_dense=False)
    reloc = ScanRelocalization(merged); wall = time.monotonic(); start = float(data.time)
    scanner = None; decision = None
    with OnboardSensorInterface(model, data) as sensors:
        for tick in range(50*60):
            if tick % 5 == 0:
                packet = sensors.capture(include_robot_mask=False)
                tracking = slam.update(packet.cameras['navigation'], packet.timestamp)
                if scanner is None:  # stowed posture measured from encoders at start
                    scanner = LocalScanSkill(ScanConfig(stowed_arm=tuple(packet.arm.position)))
                if reloc.observe(packet.cameras['navigation'], tracking, float(packet.timestamp-start)):
                    break
            decision = scanner.step(packet, tracking, float(data.time))
            if scanner.terminal:  # like the pipeline: transient statuses (e.g. tracking not yet accepted) do not end the turn
                break
            base.step(0., decision.yaw_rps)
        reloc.finish('scan_ended:'+(decision.status if decision else 'none'))
    reloc.save(output)
    result = dict(reloc.result); sim_seconds = float(data.time-start)
    origin, reference = reference_origin(memory_root, merged)
    expected = np.linalg.inv(origin) @ truth_start
    audit = dict(source=str(source), house=house[:12], task=task, memory=str(merged), reference_episode=reference,
                 accepted=result.get('accepted'), status=result.get('status'), method=result.get('localization_method'),
                 views=result.get('views'), swept_degrees=result.get('swept_degrees'),
                 scan_simulated_seconds=sim_seconds, total_wall_seconds=time.monotonic()-wall,
                 match_wall_seconds=result.get('wall_seconds'), scan_stop=decision.status if decision else None,
                 scope='private after-run score; truth never used by scan or matcher')
    if result.get('accepted'):
        est = np.asarray(result['T_map_initial_base'])
        d = yaw(est)-yaw(expected)
        audit.update(position_error_m=float(np.linalg.norm(est[:2, 3]-expected[:2, 3])),
                     yaw_error_degrees=float(abs(np.rad2deg(np.arctan2(np.sin(d), np.cos(d))))))
    if reloc.query is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(reloc.history_points); q = reloc.query
        d = tree.query(q@expected[:3, :3].T+expected[:3, 3])[0]
        audit.update(truth_fraction_within_15cm=float(np.mean(d < .15)), truth_median_distance_m=float(np.median(d)),
                     reconstructed_points=int(len(q)), estimate_fraction_within_15cm=result.get('best_fraction_within_15cm'))
        np.savez_compressed(output/'reconstruction.npz', query=q.astype(np.float32), poses=np.array(reloc.poses))
        figure(output, reloc, expected, np.asarray(result['T_map_initial_base']) if result.get('accepted') else None, audit)
    (output/'private_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    return audit


if __name__ == '__main__':
    cv2.setNumThreads(8); cv2.setRNGSeed(0)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--memory', type=Path, default=(Path(__file__).resolve().parents[1]/'runs/all_open_fixed_v4/memory/visual'), help='Visual memory root (has merged/ and episodes/)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        try: run(s, a.memory, a.output/s.name)
        except Exception as error: print(json.dumps(dict(source=str(s), error=f'{type(error).__name__}: {error}')), flush=True)
