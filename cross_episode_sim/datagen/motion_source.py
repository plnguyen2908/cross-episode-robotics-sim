"""Retarget successful contact segments, without replaying simulator states.

This is a MimicGen-style prototype, not the NVlabs implementation. A segment
is expressed relative to its final TCP target (the task's object-relative goal).
The new skill supplies that goal. A checked connector joins the current TCP to
the transformed demonstration; ordinary planning remains the fallback.
"""
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate(a, b, fraction):
    pose = np.eye(4)
    pose[:3, 3] = (1-fraction)*a[:3, 3] + fraction*b[:3, 3]
    pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(fraction).as_matrix()
    return pose


def retarget(poses, start, goal, spacing=.005, angle_spacing=.05):
    """Rigid target-frame adaptation plus a connector, at bounded pose spacing."""
    poses = np.asarray(poses)
    transformed = goal @ np.linalg.inv(poses[-1]) @ poses
    anchors = [start, *transformed]
    result = []
    for a, b in zip(anchors, anchors[1:]):
        distance = np.linalg.norm(b[:3, 3]-a[:3, 3])
        angle = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude()
        count = max(1, int(np.ceil(max(distance/spacing, angle/angle_spacing))))
        if distance < 1e-5 and angle < 1e-4:
            continue
        result.extend(interpolate(a, b, f) for f in np.linspace(0, 1, count+1)[1:])
    if not result:
        result = [goal.copy()]
    return result


class MotionSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        report = json.loads((self.path/'report.json').read_text())
        if report.get('success') is not True:
            raise ValueError(f'Motion source must be a successful physical run: {self.path}')
        trace = json.loads((self.path/'trace.json').read_text())
        # Only contact-servo stages are compatible with the consuming executor.
        contact_stages = {e['stage'] for e in report['stages'] if 'contact_method' in e}
        completions = defaultdict(list)
        for event in report['stages']:
            if 'contact_method' in event:
                completions[event['stage'], event['object']].append(event['time'])
        for times in completions.values():
            times.sort()
        self.segments = {}
        previous = None
        rows = []

        def save():
            if not rows or previous[0] not in contact_stages or 'pose correction' in previous[0]:
                return
            # The trace is 25 Hz; the contact servo advances one target per .32 s.
            # Decimate stationary samples instead of executing every recorded tick.
            chosen = rows[::8]
            if chosen[-1] is not rows[-1]:
                chosen.append(rows[-1])
            self.segments.setdefault(previous[:3], []).append(np.asarray([r['tcp'] for r in chosen]))

        for row in trace:
            stage, obj = row['stage'], row.get('active_object')
            # Multiple actuator moves may share one stage label (e.g. lowering
            # the portafilter in small increments). Keep each completed move
            # separate; stage changes alone cannot identify these boundaries.
            part = bisect_left(completions[stage, obj], row.get('time', 0)-1e-8)
            key = (stage, obj, row.get('review_phase'), part)
            if key != previous:
                save()
                rows = []
                previous = key
            rows.append(row)
        save()
        self.grasp = report.get('annotation_selection', {})

    def targets(self, controller, stage, goal):
        key = (stage, controller.object_name, getattr(controller, 'review_phase', None))
        candidates = self.segments.get(key, [])
        if not candidates:
            return None
        start = controller.tcp().copy()
        def entry_distance(segment):
            entry = goal @ np.linalg.inv(segment[-1]) @ segment[0]
            return np.linalg.norm(entry[:3, 3]-start[:3, 3])
        selected = min(candidates, key=entry_distance)
        if entry_distance(selected) > .15:
            return None  # Different approach/context: use the normal planner.
        return retarget(selected, start, goal)


def source_for(controller):
    """Lazy per-controller source; opt-in only, including worker subprocesses."""
    import os
    path = os.environ.get('CES_MOTION_SOURCE')
    if not path:
        return None
    if not hasattr(controller, '_motion_source'):
        controller._motion_source = MotionSource(path)
        controller.report['motion_source'] = str(controller._motion_source.path)
    return controller._motion_source
