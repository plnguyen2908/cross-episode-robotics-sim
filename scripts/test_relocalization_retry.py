"""Localization with one retry: first 360-degree scan, and if it misses the memory gate, a second scan.
Private score: position error of the accepted localization against the true start pose (map frame).
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_nav_replay import MEM, START_GATE, Runner, body_world  # noqa: E402


def run(source, out):
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    from cross_episode_sim.evaluation.text_memory import house_of
    from cross_episode_sim.navigation.local_scan import LocalScanSkill, ScanConfig
    from cross_episode_sim.navigation.visual_relocalization import ScanRelocalization, memory_directory
    from cross_episode_sim.sensors.onboard import OnboardSensorInterface
    from cross_episode_sim.slam import create_mapper
    out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    house, _ = house_of(source); house = house[:12]; task = json.loads((source/'config.json').read_text())['task']
    model, data, _ = load_saved_scene(source); merged = memory_directory(MEM/'visual_tour_root', house, task)
    W0 = np.linalg.inv(np.load(MEM/'visual_tours'/house/'private_start_pose.npy')); slam = create_mapper(fuse_dense=False)
    report = dict(source=str(source), house=house, tries=[])
    with OnboardSensorInterface(model, data) as sensors:
        R = Runner(model, data, sensors, slam); R.sense(); s0 = float(data.time)
        open_dirs = []
        for k in range(2):
            if k == 1 and open_dirs:  # new viewpoint: most open direction seen in the first scan's depth, up to 1 m
                yaw, depth = max(open_dirs, key=lambda x: x[1]); move = min(1., max(0., depth-1.))
                for _ in range(int(8/.02)):
                    T = np.asarray(R.tracking.T_map_base) if R.tracking is not None and R.tracking.accepted else None
                    if T is None:R.tick(0., 0.); continue
                    err = math.atan2(math.sin(yaw-math.atan2(T[1, 0], T[0, 0])), math.cos(yaw-math.atan2(T[1, 0], T[0, 0])))
                    if abs(err) < .05:break
                    R.tick(0., float(np.clip(2.*err, -1., 1.)))
                p0 = np.asarray(R.tracking.T_map_base)[:2, 3].copy() if R.tracking is not None and R.tracking.accepted else None
                for _ in range(int(6/.02)):
                    if p0 is None or move <= 0:break
                    if R.tracking is not None and R.tracking.accepted and np.linalg.norm(np.asarray(R.tracking.T_map_base)[:2, 3]-p0) >= move:break
                    ahead = R.packet.cameras['navigation'].depth_m; hh, ww = ahead.shape; ahead = ahead[hh//4:3*hh//4, ww//3:2*ww//3]
                    if np.isfinite(ahead).any() and np.nanpercentile(ahead, 5) < 1.:break  # keep >= 1 m ahead: room for the scan's turning circle
                    R.tick(.3, 0.)
                for _ in range(25):R.tick(0., 0.)
                report['move_m'] = round(move, 2)
            reloc = ScanRelocalization(merged); sc = LocalScanSkill(ScanConfig(stowed_arm=tuple(R.packet.arm.position))); t_s = float(data.time)
            while float(data.time)-t_s < 40.:
                if k == 0 and R.tick_n % 5 == 0 and R.tracking is not None and R.tracking.accepted:
                    dep = R.packet.cameras['navigation'].depth_m; h, w_ = dep.shape; band = dep[:h//2, w_//3:2*w_//3]  # upper half: walls and openings at range (the mast camera tilts down)
                    T = np.asarray(R.tracking.T_map_base); open_dirs.append((math.atan2(T[1, 0], T[0, 0]), float(np.nanmedian(band)) if np.isfinite(band).any() else 0.))
                if R.tick_n % 5 == 0 and reloc.observe(R.packet.cameras['navigation'], R.tracking, float(R.packet.timestamp-s0)):break
                dec = sc.step(R.packet, R.tracking, float(data.time))
                if sc.terminal:break
                R.tick(0., dec.yaw_rps)
            res = reloc.finish('scan_ended') if not reloc.done else reloc.result
            row = {k: res.get(k) for k in ('status', 'fine_fit_mean_capped_distance_m', 'views')}
            if res.get('T_memory_from_slam') is not None and R.tracking is not None and R.tracking.accepted:
                est = np.asarray(res['T_memory_from_slam'])@np.asarray(R.tracking.T_map_base); true = W0@body_world(model, data)
                row['private_error_cm'] = round(float(np.linalg.norm(est[:2, 3]-true[:2, 3]))*100, 1)
            row['confident'] = bool(res.get('accepted') and (res.get('fine_fit_mean_capped_distance_m') or 1) <= START_GATE)
            report['tries'].append(row)
            if row['confident']:break
    report['use_memory'] = report['tries'][-1]['confident']; report['wall_seconds'] = round(time.monotonic()-t0, 1)
    (out/'report.json').write_text(json.dumps(report, indent=1)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for s in a.source:
        try:print(json.dumps(run(s, a.output/f'{s.parent.name}_{s.name}')), flush=True)
        except Exception as e:print(json.dumps(dict(source=str(s), error=f'{type(e).__name__}: {e}')), flush=True)
