"""Gravity-aligned global registration of an observed local room submap.

No simulator inputs, semantic room truth, known query position, or RGB texture.
Search planar translation/yaw over height-separated geometry, then refine in
metric 3D. Missing observations are not fabricated. Symmetric solutions remain
ambiguous rather than being resolved by a hidden room label.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from ..perception.surfaces import project_mask


@dataclass(frozen=True)
class RoomGeometryConfig:
    resolution: float = .10
    yaw_step_degrees: float = 5.
    coarse_sigma: float = .15
    max_candidates: int = 24
    min_agreement: float = .70
    max_median_distance: float = .08
    ambiguity_score_margin: float = .03
    history_voxel_m: float = .06
    query_voxel_m: float = .10


def compact(points, size=.06):
    p = np.asarray(points, dtype=np.float64)
    p = p[np.isfinite(p).all(1)]
    _, ids = np.unique(np.floor(p/size).astype(np.int64), axis=0, return_index=True)
    return p[ids]


def reconstruct_scan(cameras, relative_poses):
    """Fuse a rotation scan in its initial-base frame using tracked motion."""
    if len(cameras) != len(relative_poses):
        raise ValueError('A tracked pose is required for each camera view')
    clouds = []
    for c, pose in zip(cameras, relative_poses):
        if not np.isfinite(pose).all():
            continue
        valid = c.valid_depth & np.isfinite(c.depth_m) & (c.depth_m > .65) & (c.depth_m < 6.)
        if c.robot_mask is not None:
            valid &= ~c.robot_mask
        sample = np.zeros_like(valid); sample[::4, ::4] = True
        p = project_mask(c.depth_m, valid & sample, c.K, np.asarray(pose)@c.T_base_optical)
        clouds.append(p[(p[:, 2] > .2) & (p[:, 2] < 2.5)])
    return compact(np.concatenate(clouds)) if clouds else np.empty((0, 3))


def yaw_difference(a, b):
    return abs(np.arctan2(np.sin(a-b), np.cos(a-b)))


class RoomGeometryLocalizer:
    """Search a frozen measured map; construction is reusable across queries."""
    bands = ((.2, .6), (.6, 1.1), (1.1, 1.85), (1.85, 2.5))

    def __init__(self, history_points, config=None):
        self.config = config or RoomGeometryConfig()
        cfg = self.config
        self.history = compact(history_points, cfg.history_voxel_m)
        self.history = self.history[(self.history[:, 2] > .2) & (self.history[:, 2] < 2.5)]
        if len(self.history) < 100:
            raise ValueError('Insufficient observed history geometry')
        self.tree = cKDTree(self.history)
        self.origin = np.floor(self.history[:, :2].min(0)/cfg.resolution)*cfg.resolution-.5
        shape = np.ceil((self.history[:, :2].max(0)-self.origin+.5)/cfg.resolution).astype(int)+1
        self.fields = []
        for low, high in self.bands:
            p = self.history[(self.history[:, 2] >= low) & (self.history[:, 2] < high)]
            occupied = np.zeros(shape[::-1], dtype=bool)
            indices = np.floor((p[:, :2]-self.origin)/cfg.resolution).astype(int)
            if len(indices):
                occupied[indices[:, 1], indices[:, 0]] = True
                distance = distance_transform_edt(~occupied)*cfg.resolution
                field = np.exp(-.5*(distance/cfg.coarse_sigma)**2)
            else:
                field = np.zeros_like(occupied, dtype=float)
            self.fields.append(field.astype(np.float32))

    def _search(self, query):
        cfg = self.config
        proposals = []
        for degrees in np.arange(0, 360, cfg.yaw_step_degrees):
            yaw = np.deg2rad(degrees)
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            xy = query[:, :2]@R.T
            qmin = np.floor(xy.min(0)/cfg.resolution).astype(int)
            indices = np.floor(xy/cfg.resolution).astype(int)-qmin
            shape = indices.max(0)+1
            score = None; active = 0
            for field, (low, high) in zip(self.fields, self.bands):
                selected = indices[(query[:, 2] >= low) & (query[:, 2] < high)]
                if len(selected) < 10 or not field.any():
                    continue
                template = np.zeros(shape[::-1], dtype=np.float32)
                template[selected[:, 1], selected[:, 0]] = 1
                part = fftconvolve(field, template[::-1, ::-1], mode='full')/template.sum()
                score = part if score is None else score+part
                active += 1
            if score is None:
                continue
            score /= active
            # Keep spatially separated maxima at every heading, not just one.
            for _ in range(8):
                row, col = np.unravel_index(np.argmax(score), score.shape)
                value = float(score[row, col])
                shift = np.array([col, row])-(shape-1)
                translation = self.origin+(shift-qmin)*cfg.resolution
                proposals.append(dict(x=float(translation[0]), y=float(translation[1]),
                                      yaw=float(yaw), coarse_score=value))
                radius = int(np.ceil(.5/cfg.resolution))
                score[max(0, row-radius):row+radius+1, max(0, col-radius):col+radius+1] = -1
        proposals.sort(key=lambda x:x['coarse_score'], reverse=True)
        chosen = []
        for p in proposals:
            if any(np.hypot(p['x']-q['x'], p['y']-q['y']) < .5 and
                   yaw_difference(p['yaw'], q['yaw']) < np.deg2rad(20) for q in chosen):
                continue
            chosen.append(p)
            if len(chosen) >= cfg.max_candidates:
                break
        return chosen

    def _refine(self, query, proposal):
        full_query = query
        if len(query) > 1200:
            query = query[np.linspace(0, len(query)-1, 1200, dtype=int)]
        yaw = proposal['yaw']; t = np.array([proposal['x'], proposal['y']])
        for _ in range(30):
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            moved = query.copy(); moved[:, :2] = query[:, :2]@R.T+t
            distances, ids = self.tree.query(moved)
            keep = distances < .3
            if keep.sum() < 50:
                break
            src, dst = query[keep, :2], self.history[ids[keep], :2]
            # Robust weights reduce the influence of unmatched/moved objects.
            weights = np.minimum(1., .08/np.maximum(distances[keep], 1e-6))
            weights /= weights.sum()
            a, b = np.sum(src*weights[:, None], axis=0), np.sum(dst*weights[:, None], axis=0)
            U, _, Vt = np.linalg.svd((src-a).T@((dst-b)*weights[:, None]))
            correction = np.eye(2); correction[-1, -1] = np.linalg.det(Vt.T@U.T)
            newR = Vt.T@correction@U.T
            newt = b-newR@a; newyaw = float(np.arctan2(newR[1, 0], newR[0, 0]))
            done = np.linalg.norm(newt-t) < 1e-4 and yaw_difference(newyaw, yaw) < 1e-4
            yaw, t = newyaw, newt
            if done:
                break
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        query = full_query
        moved = query.copy(); moved[:, :2] = query[:, :2]@R.T+t
        distances = self.tree.query(moved)[0]
        T = np.eye(4); T[:2, :2] = R; T[:2, 3] = t
        fraction = float(np.mean(distances < .15)); median = float(np.median(distances))
        return dict(T_map_initial_base=T.tolist(), fraction_within_15cm=fraction,
                    median_distance_m=median, coarse_score=proposal['coarse_score'],
                    accepted=fraction >= self.config.min_agreement and median < self.config.max_median_distance)

    def localize(self, query):
        start = time.monotonic()
        query = compact(query, self.config.query_voxel_m)
        query = query[(query[:, 2] > .2) & (query[:, 2] < 2.5)]
        if len(query) < 100:
            return dict(accepted=False, status='insufficient_geometry', T_map_initial_base=None)
        proposals = self._search(query)
        search_seconds = time.monotonic()-start
        candidates = [self._refine(query, p) for p in proposals]
        candidates.sort(key=lambda c:(c['fraction_within_15cm'], -c['median_distance_m']), reverse=True)
        if not candidates:
            return dict(accepted=False, status='insufficient_geometry', T_map_initial_base=None)
        best = candidates[0]; A = np.array(best['T_map_initial_base'])
        alternatives = []
        for c in candidates[1:]:
            if not c['accepted'] or c['fraction_within_15cm'] < best['fraction_within_15cm']-self.config.ambiguity_score_margin:
                continue
            B = np.array(c['T_map_initial_base'])
            if np.linalg.norm(A[:2, 3]-B[:2, 3]) > .3 or yaw_difference(np.arctan2(A[1, 0], A[0, 0]), np.arctan2(B[1, 0], B[0, 0])) > np.deg2rad(15):
                alternatives.append(c)
        accepted = best['accepted'] and not alternatives
        return dict(accepted=accepted, status='localized' if accepted else 'ambiguous' if alternatives else 'no_consistent_match',
            T_map_initial_base=best['T_map_initial_base'] if accepted else None,
            candidates=candidates, alternate_candidates=len(alternatives), query_points=len(query),
            search_seconds=search_seconds, wall_seconds=time.monotonic()-start,
            room_assignment=None,
            scope='Observed submap global x/y/yaw search and planar metric refinement. No room truth. Semantic room assignment requires separately observed room regions.')


def reconstruct_history(relocalizer):
    """Build a reusable reference from history RGB-D and estimated key poses."""
    from contextlib import ExitStack
    import h5py
    points = []
    with ExitStack() as stack:
        streams = {}
        for episode, key, calibration in relocalizer.references:
            if episode not in streams:
                streams[episode] = stack.enter_context(h5py.File(relocalizer.root/'streams'/episode/'sensors.h5'))
            f = streams[episode]; i = int(key.index)
            depth = f['depth_m'][i]
            valid = np.isfinite(depth) & (depth > .65) & (depth < 6.) & ~f['robot_mask'][i].astype(bool)
            sample = np.zeros_like(valid); sample[::4, ::4] = True
            p = project_mask(depth, valid & sample, np.asarray(calibration['K']), key.pose)
            points.append(p[(p[:, 2] > .2) & (p[:, 2] < 2.5)])
    return compact(np.concatenate(points))
