"""Timestamped historical commands, kept separate from video-only captions."""
import json
from pathlib import Path

import numpy as np


def action_slice(times, start, end):
    """Half-open command-start interval [start, end); no interpolation."""
    times = np.asarray(times)
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError('Invalid segment interval')
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Commands require strictly increasing finite timestamps')
    return int(np.searchsorted(times, start)), int(np.searchsorted(times, end))


def rebase_planar_commands(ctrl, actuator_names, transform):
    """Rebase absolute base targets using a sensor-estimated rigid transform.

    This is command geometry, not a measurement of the historical base path.
    Arm/gripper commands and heading continuity are preserved exactly.
    """
    T = np.asarray(transform, dtype=float)
    if T.shape != (4,4) or not np.isfinite(T).all():
        raise ValueError('A finite 4x4 recording-to-execution transform is required')
    if (not np.allclose(T[3], [0,0,0,1]) or not np.allclose(T[2], [0,0,1,0])
            or not np.allclose(T[:2,2], 0) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3))
            or not np.isclose(np.linalg.det(T[:3,:3]),1)):
        raise ValueError('Expected a planar rigid transform')
    indices = []
    for suffix in ('base_x_act','base_y_act','base_theta_act'):
        matches = [i for i,name in enumerate(actuator_names) if name.rsplit('/',1)[-1] == suffix]
        if len(matches) != 1: raise ValueError('Unsupported native base command schema')
        indices.append(matches[0])
    result = np.asarray(ctrl).copy()
    x,y,yaw = indices
    result[:,[x,y]] = result[:,[x,y]]@T[:2,:2].T+T[:2,3]
    result[:,yaw] += np.arctan2(T[1,0],T[0,0])
    return result


class ActionMemory:
    """Load exported commands and retrieve exact recorded skill segments."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        if self.manifest['schema_version'] != 'action_memory.v1':
            raise ValueError('Unsupported action-memory version')
        if self.manifest.get('status') != 'complete':
            raise ValueError('Action-memory export is incomplete')
        self.episodes = {
            row['episode_id']: json.loads((self.directory/row['metadata']).read_text())
            for row in self.manifest['episodes']}

    def candidates(self, skill, arguments, *, episode_ids):
        """Caller scopes histories to the current house/task; no global fallback."""
        result = []
        for episode in episode_ids:
            record = self.episodes[episode]
            for segment in record['skills']:
                if segment['skill'] != skill or not segment['recorded_success'] or not segment.get('replayable', True):
                    continue
                # Reject omitted semantic arguments too (e.g. carried target).
                saved = {k: v for k, v in segment['arguments'].items() if k != 'dock'}
                requested = {k: v for k, v in arguments.items() if k != 'dock'}
                if saved != requested:
                    continue
                if 'dock' in arguments and segment['arguments'].get('dock', '0') != arguments['dock']:
                    continue
                result.append(dict(episode_id=episode, **segment))
        return sorted(result, key=lambda row: (row['duration_seconds'], row['episode_id'], row['segment_id']))

    def trajectory(self, episode_id, segment):
        record = self.episodes[episode_id]
        first, stop = segment['action_range']
        if stop <= first:
            raise ValueError('Segment contains no recorded actions')
        with np.load(self.directory/record['commands_file'], allow_pickle=False) as raw:
            if not 0 <= first < stop <= len(raw['time']):
                raise ValueError('Invalid action range')
            arrays = {key: raw[key][first:stop].copy() for key in ('time','ctrl','forcerange','stage')}
        arrays['relative_time'] = arrays['time']-arrays['time'][0]
        arrays['step_seconds'] = np.diff(np.r_[arrays['time'], segment['execution_end_time']])
        if np.any(arrays['step_seconds'] <= 0):
            raise ValueError('Invalid final command duration')
        return arrays
