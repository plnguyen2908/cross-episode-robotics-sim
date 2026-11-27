"""Exploration-style navigation map of a house from its planned tour.

The same ObservedNavigationMap the exploration stage builds (same MapConfig:
5 cm cells, 5 cm height bins, pixel stride 4, growing extent), fused from the
tour's onboard depth frames (map_frames.h5, 5 Hz) at the tour's SLAM poses after
straightening (straighten.json piece corrections). Saved in the exploration
format (navigation_map.npz, floor_observations.npz, occupied_sources.npz,
summary.json) under merged_tour/<house>/navigation_map/, plus map.png drawn
with the exploration plotter. Onboard observations and estimated poses only.
"""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_episode_sim.navigation.observed_map import MapConfig, ObservedNavigationMap  # noqa: E402
from cross_episode_sim.sensors.onboard import RGBDCamera, depth_validity  # noqa: E402


def piece_corrections(merged, episode):
    path = merged/'straighten.json'
    if not path.exists():
        return 30., {}
    log = json.loads(path.read_text())
    table = {e['piece']: np.asarray(e['correction_matrix']) for e in log['pieces']
             if e.get('episode') == episode and 'correction_matrix' in e}
    return float(log['segment_seconds']), table


def build(memory, house):
    vis = memory/'visual'; merged = vis/'merged_tour'/house
    graph = json.loads((merged/'graph.json').read_text())
    grid = ObservedNavigationMap(MapConfig(extent_m=12., resolution=.05, height_resolution=.05, pixel_stride=4))
    used = 0; offset = 0.; any_table = False; segment = 30.
    for eid, meta in graph['episodes'].items():  # one episode per tour (loop); all in the map frame
        G = np.asarray(meta['transform'])
        traj = np.load(vis/'episodes'/eid/'trajectory.npz'); poses, accepted, times = traj['poses'], traj['accepted'], traj['time']
        cal = json.loads((vis/'episodes'/eid/'calibration.json').read_text()); K = np.asarray(cal['K']); mount = np.asarray(cal['mount'])
        segment, table = piece_corrections(merged, eid); known = sorted(table); any_table |= bool(table)
        with h5py.File(vis/'streams'/eid/'map_frames.h5') as f:
            for n, j in enumerate(f['index'][:]):
                j = int(j)
                if not accepted[j]:continue
                k = int((j/20.)//segment)
                C = table[max([q for q in known if q <= k], default=known[0])] if known else np.eye(3)
                C4 = np.eye(4); C4[:2, :2] = C[:2, :2]; C4[:2, 3] = C[:2, 2]
                depth = f['depth_m'][n].astype(np.float32); mask = f['robot_mask'][n].astype(bool)
                camera = RGBDCamera('navigation', np.zeros((*depth.shape, 3), np.uint8), depth, depth_validity(depth), K,
                                    np.zeros(5), mount, mask, depth)
                grid.update(camera, SimpleNamespace(timestamp=offset+float(times[j]), accepted=True, T_map_base=C4@G@poses[j], status='accepted'))
                used += 1
        offset += float(times[-1])+1.  # loops restart their clock; keep observation times increasing
    table = any_table
    out = merged/'navigation_map'
    summary = grid.save(out)
    from scripts.demo_local_scan import plot_grid
    plot_grid(out, f"{house}: planned-tour navigation map ({'straightened' if table else 'raw'} onboard SLAM poses)")
    (out/'build.json').write_text(json.dumps(dict(frames_used=used, straightened=bool(table), segment_seconds=segment,
                                                  config='exploration MapConfig: 5 cm, pixel_stride 4'), indent=1))
    return dict(house=house, frames=used, **{k: summary[k] for k in ('floor_area_m2',) if k in summary})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory', type=Path, default=Path('/nobackup/le/cross-episode-robotics-sim/runs/all_open_fixed_v4/memory'))
    p.add_argument('--houses', nargs='+', required=True)
    a = p.parse_args()
    for h in a.houses:
        print(json.dumps(build(a.memory, h)), flush=True)
