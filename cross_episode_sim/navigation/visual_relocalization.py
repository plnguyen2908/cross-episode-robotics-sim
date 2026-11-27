"""Optional startup relocalization against a saved visual memory.

During the robot's first in-place turn, keep onboard RGB-D views about every
10 degrees of *estimated* (SLAM) heading. Once a full turn is seen, fuse them
into a 3D reconstruction of the current room (scan-start frame), then align that
geometry with the history map's room geometry: correlative x/y/yaw search, then
planar ICP (docs/history_visual_route_memory.md in qwen-capx). Report where the
robot is in that map's frame, or stay unlocalized when ambiguous.
method='image_then_geometry' keeps the older image-first matcher.

Inputs are onboard images, depth, camera calibration and SLAM poses only. No
simulator pose, room label or object truth is read. Motion is never changed:
on failure the robot just keeps exploring without a memory pose.
"""
import dataclasses
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..sensors.onboard import RGBDCamera


def memory_directory(path, house_id=None, task=None):
    """A merged map folder (has graph.json), or a memory root + house/task.

    A house map (merged/<house12>, all tasks' best-tracked history) is preferred
    over the older house+task map (merged/<house12>_<task>).
    """
    path = Path(path)
    if (path/'graph.json').is_file():
        return path
    if house_id and task:
        for candidate in (path/'merged'/house_id[:12], path/'merged'/f'{house_id[:12]}_{task}', path/f'{house_id[:12]}_{task}'):
            if (candidate/'graph.json').is_file():
                return candidate
    raise FileNotFoundError(f'No merged visual memory at {path} for house={house_id} task={task}')


def memory_calibration(merged):
    """K and mount of the memory's camera, from its first reference episode."""
    merged = Path(merged)
    graph = json.loads((merged/'graph.json').read_text())
    episode = next(iter(graph['episodes']))
    calibration = json.loads((merged.parent.parent/'episodes'/episode/'calibration.json').read_text())
    return np.asarray(calibration['K'], float), np.asarray(calibration['mount'], float)


# Keep the visual memory only when the onboard scan fits the memory map this tightly (mean scan-to-map
# distance, capped at 5 cm). Calibrated on 200 test-split starts: 144 pass, none of them over 10 cm off;
# every other start explores normally.
MEMORY_FIT_GATE = .030
# user 2026-11-26 (same-layout eval: 7 of 20 relocalizations at 3.03-3.95 cm were refused and their full live explores
# of 4-5 room houses ran out of time): also accept up to 3.5 cm when nearly the whole scan agrees with the map
MEMORY_FIT_GATE_WIDE, MEMORY_WIDE_MIN_FRACTION = .035, .95


def decide(report):
    """use_memory only for an accepted match whose fine fit passes MEMORY_FIT_GATE (or MEMORY_FIT_GATE_WIDE when at least
    MEMORY_WIDE_MIN_FRACTION of the scan lies within 15 cm of the memory map); otherwise explore."""
    fit = report.get('fine_fit_mean_capped_distance_m'); frac = report.get('best_fraction_within_15cm')
    report['memory_fit_gate_m'] = MEMORY_FIT_GATE
    report['memory_fit_gate_wide_m'] = MEMORY_FIT_GATE_WIDE; report['memory_wide_min_fraction'] = MEMORY_WIDE_MIN_FRACTION
    tight = fit is not None and fit <= MEMORY_FIT_GATE
    wide = fit is not None and fit <= MEMORY_FIT_GATE_WIDE and frac is not None and float(frac) >= MEMORY_WIDE_MIN_FRACTION
    report['use_memory'] = bool(report.get('accepted') and (tight or wide))
    if report['use_memory'] and not tight:report['memory_gate'] = 'wide'
    report['decision'] = 'use memory' if report['use_memory'] else 'explore (memory not confident)'
    return report


def resize_camera(camera, K_target):
    """Resample an onboard view to the memory camera's size (same field of view).

    Uses the pixel-centre convention, so a 640x480 render scaled by 0.9375
    lands on the 600x450 memory intrinsics. Refuses if the result disagrees.
    """
    width, height = int(round(2*K_target[0, 2]+1)), int(round(2*K_target[1, 2]+1))
    h, w = camera.depth_m.shape
    if (w, h) == (width, height):
        K = camera.K
        resized = camera
    else:
        sx, sy = width/w, height/h
        K = np.array(camera.K, float)
        K[0, 0] *= sx; K[1, 1] *= sy
        K[0, 2] = (K[0, 2]+.5)*sx-.5; K[1, 2] = (K[1, 2]+.5)*sy-.5
        near = lambda x: cv2.resize(x.astype(np.uint8) if x.dtype == bool else x, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = near(camera.depth_m)
        valid = near(camera.valid_depth).astype(bool)
        mask = None if camera.robot_mask is None else near(camera.robot_mask).astype(bool)
        resized = RGBDCamera(camera.frame_id, cv2.resize(camera.rgb, (width, height), interpolation=cv2.INTER_AREA),
                             depth, valid, K, camera.distortion, camera.T_base_optical, mask, None, camera.robot_bounds_base)
    if not np.allclose(K, K_target, rtol=1e-3, atol=.05):
        raise ValueError(f'Onboard camera K {np.round(K, 3).tolist()} cannot match memory K {np.round(K_target, 3).tolist()}')
    # Exact memory intrinsics after the tolerance check (sub-0.05 px rounding).
    return dataclasses.replace(resized, K=np.array(K_target, float))


def _prepare(query, history):
    from scipy.spatial import cKDTree
    from ..mapping.room_geometry_localization import compact
    tree = cKDTree(compact(history, .03))
    q = compact(query, .05); q = q[(q[:, 2] > .2) & (q[:, 2] < 2.5)]
    return tree, q


def fine_refine(query, history, T, cutoffs=(.15, .10, .07, .05, .05, .04, .04, .03), inner=5, prepared=None):
    """Final planar ICP (x, y, yaw) with a shrinking match distance, from an accepted pose.

    The coarse search scores fit within 15 cm, which leaves a degree or two of
    heading free; tightening to 3 cm pins it down. Uses only the scan and map.
    """
    tree, q = prepared or _prepare(query, history)
    T = np.array(T, float)
    for cutoff in cutoffs:
        for _ in range(inner):
            p = q@T[:3, :3].T+T[:3, 3]; d, j = tree.query(p); m = d < cutoff
            if m.sum() < 200:
                return T
            a, b = p[m, :2], tree.data[j[m], :2]; ca, cb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a-ca).T@(b-cb)); R = Vt.T@U.T
            if np.linalg.det(R) < 0:
                Vt[1] *= -1; R = Vt.T@U.T
            D = np.eye(4); D[:2, :2] = R; D[:2, 3] = cb-R@ca; T = D@T
    return T


def multistart_refine(query, history, T, shifts=(-.2, -.1, 0., .1, .2), turns_deg=(-1., 0., 1.), keep=5):
    """fine_refine from nearby starts around T; keep the pose whose scan fits the map tightest.

    Fixes a refinement that settles in a neighbouring basin (seen: 16 cm -> 0.1 cm).
    All 75 starts are first scored cheaply after a short ICP; only the best --keep
    (and T itself) get the full refinement. Score: mean scan-to-map distance capped
    at 5 cm. Scan and map only.
    """
    import itertools
    prepared = _prepare(query, history); tree, q = prepared
    sub = q[::4]
    score = lambda P, pts: float(np.mean(np.minimum(tree.query(pts@P[:3, :3].T+P[:3, 3])[0], .05)))
    T = np.array(T, float); quick = []
    for dx, dy, turn in itertools.product(shifts, shifts, turns_deg):
        D = np.eye(4); c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn)); D[:2, :2] = [[c, -s_], [s_, c]]
        start = T@D; start[:2, 3] += (dx, dy)
        F = fine_refine(None, None, start, cutoffs=(.15, .07, .04), inner=2, prepared=(tree, sub))
        quick.append((score(F, sub), F))
    quick.sort(key=lambda x: x[0])
    best = None
    for _, S0 in [(0, T)]+quick[:keep]:
        F = fine_refine(None, None, S0, prepared=prepared); v = score(F, q)
        if best is None or v < best[0]:
            best = (v, F)
    return best[1], best[0]


class ScanRelocalization:
    """Collect views over the first full turn, then match once.

    observe(...) at the sensor rate; it returns True when matching has run.
    The result is in .result and written to <output>/visual_relocalization.json.
    """
    def __init__(self, merged, spacing_degrees=10., full_turn_degrees=350., min_view_angle_degrees=15., method='geometry'):
        if method not in ('geometry', 'image_then_geometry'):
            raise ValueError('method must be geometry or image_then_geometry')
        self.method = method; self.query = None; self.history_points = None
        self.merged = Path(merged)
        self.K, self.mount = memory_calibration(self.merged)
        self.spacing = math.radians(spacing_degrees)
        self.full_turn = math.radians(full_turn_degrees)
        self.min_view_angle = min_view_angle_degrees
        self.cameras, self.poses, self.stamps = [], [], []
        self.first_pose = None; self.previous_yaw = None; self.last_kept_yaw = None
        self.swept = 0.; self.max_swept = 0.; self.min_swept = 0.
        self.done = False; self.result = None; self.error = None

    @staticmethod
    def _yaw(T):
        return math.atan2(T[1, 0], T[0, 0])

    def observe(self, camera, tracking, timestamp=None):
        if self.done:
            return False
        if not tracking.accepted:
            return False
        T = np.asarray(tracking.T_map_base, float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            return False
        if self.first_pose is None:
            self.first_pose = T.copy()
        yaw = self._yaw(T)
        if self.previous_yaw is not None:
            self.swept += math.atan2(math.sin(yaw-self.previous_yaw), math.cos(yaw-self.previous_yaw))
        self.previous_yaw = yaw
        self.max_swept = max(self.max_swept, self.swept); self.min_swept = min(self.min_swept, self.swept)
        if self.last_kept_yaw is None or abs(self.swept-self.last_kept_yaw) >= self.spacing-1e-6:
            try:
                self.cameras.append(resize_camera(camera, self.K))
            except ValueError as error:
                self.error = str(error); self.done = True
                self.result = dict(accepted=False, status='camera_calibration_mismatch', reason=self.error)
                return True
            self.poses.append(np.linalg.inv(self.first_pose) @ T)
            self.stamps.append(timestamp); self.last_kept_yaw = self.swept
        if self.max_swept-self.min_swept >= self.full_turn:
            self.finish()
            return True
        return False

    def finish(self, reason='full_turn'):
        """Match the views gathered so far (also called when the turn ends early)."""
        if self.done:
            return self.result
        self.done = True
        base = dict(memory_directory=str(self.merged), trigger=reason, views=len(self.cameras),
                    swept_degrees=float(math.degrees(self.max_swept-self.min_swept)),
                    frame='memory map frame (first history episode start pose)')
        if len(self.cameras) < 3:
            self.result = dict(base, accepted=False, status='too_few_scan_views')
            return self.result
        if self.method == 'geometry':
            return self._finish_geometry(base)
        from ..mapping.history_localization import HistoryRelocalizer
        try:
            localizer = HistoryRelocalizer(self.merged.parent.parent, self.merged)
            ids = list(range(len(self.cameras)))
            report, _ = localizer.localize(self.cameras, ids, relative_poses=np.array(self.poses),
                                           min_view_angle_degrees=self.min_view_angle)
        except Exception as error:  # a broken memory must not stop exploration
            self.result = dict(base, accepted=False, status='matching_error', reason=f'{type(error).__name__}: {error}')
            return self.result
        report = {k: v for k, v in report.items() if k not in ('matches', 'image_result')}
        report.update(base, query_timestamps=self.stamps)
        if report.get('accepted'):
            T_memory_scan = np.asarray(report['T_map_initial_base'])
            # SLAM map -> memory map; the rest of the episode can reuse it.
            report['T_memory_from_slam'] = (T_memory_scan @ np.linalg.inv(self.first_pose)).tolist()
            current = np.asarray(report.get('T_map_current_base', T_memory_scan))
            report['position_in_memory'] = current[:2, 3].tolist()
            report['yaw_in_memory_degrees'] = float(math.degrees(self._yaw(current)))
        self.result = decide(report)
        return self.result

    def history_geometry(self):
        """History room geometry in the map frame (saved per map; else rebuilt from history streams)."""
        if self.history_points is None:
            saved = self.merged/'history_geometry.npz'
            if saved.is_file():
                with np.load(saved) as f:
                    if list(f['episodes']) != list(json.loads((self.merged/'graph.json').read_text())['episodes']):
                        raise ValueError(f'{saved}: built from other episodes than graph.json')
                    self.history_points = f['points'].astype(np.float64)
            else:
                from ..mapping.history_localization import HistoryRelocalizer
                from ..mapping.room_geometry_localization import reconstruct_history
                self.history_points = reconstruct_history(HistoryRelocalizer(self.merged.parent.parent, self.merged))
        return self.history_points

    def _finish_geometry(self, base):
        import time
        from ..mapping.room_geometry_localization import RoomGeometryLocalizer, reconstruct_scan
        start = time.monotonic()
        try:
            history = self.history_geometry()
            self.query = reconstruct_scan(self.cameras, np.array(self.poses))
            report = RoomGeometryLocalizer(history).localize(self.query)
        except Exception as error:  # a broken memory must not stop exploration
            self.result = dict(base, accepted=False, status='matching_error', reason=f'{type(error).__name__}: {error}')
            return self.result
        best = (report.get('candidates') or [{}])[0]
        report = {k: v for k, v in report.items() if k != 'candidates'}
        report.update(base, localization_method='room_geometry', query_timestamps=self.stamps,
                      reconstructed_points=int(len(self.query)), history_points=int(len(history)),
                      best_fraction_within_15cm=best.get('fraction_within_15cm'), best_median_distance_m=best.get('median_distance_m'),
                      total_wall_seconds=time.monotonic()-start)
        if report.get('accepted'):
            coarse = np.asarray(report['T_map_initial_base'])
            T_memory_scan, fit_score = multistart_refine(self.query, history, coarse)
            report['fine_fit_mean_capped_distance_m'] = fit_score
            report['T_map_initial_base_coarse'] = coarse.tolist(); report['T_map_initial_base'] = T_memory_scan.tolist()
            report['fine_refinement_cm'] = float(np.linalg.norm(T_memory_scan[:2, 3]-coarse[:2, 3])*100)
            current = T_memory_scan @ self.poses[-1]
            report['T_map_current_base'] = current.tolist()
            report['T_memory_from_slam'] = (T_memory_scan @ np.linalg.inv(self.first_pose)).tolist()
            report['position_in_memory'] = current[:2, 3].tolist()
            report['yaw_in_memory_degrees'] = float(math.degrees(self._yaw(current)))
        self.result = decide(report)
        return self.result

    def save(self, output):
        if self.result is not None:
            Path(output, 'visual_relocalization.json').write_text(json.dumps(self.result, indent=2, default=float))
