"""Remove stationary holds from a derived state trace, retaining source timing."""
import json
from pathlib import Path

import numpy as np


def trim_stationary_holds(trace, *, tolerance=1e-4, minimum_pause=.24, keep_seconds=.04):
    """Keep motion and boundaries; compress only whole-world stationary spans.

    The tolerance bounds the TOTAL range of every qpos component over a span,
    not just its per-frame difference. Thus slow accumulating motion is not
    mistaken for a pause. qpos includes robot, fingers, objects and fixtures.
    Output is an edited observation trace, not a newly validated control rollout.
    """
    if tolerance < 0 or not 0 <= keep_seconds < minimum_pause:
        raise ValueError('Require nonnegative tolerance and 0 <= keep < minimum pause')
    if not trace:
        return [], dict(input_frames=0, output_frames=0, removed_seconds=0., spans=[])
    times = np.asarray([r['time'] for r in trace], dtype=float)
    qpos = np.asarray([r['qpos'] for r in trace], dtype=float)
    if (not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0)
            or qpos.ndim != 2 or not np.all(np.isfinite(qpos))):
        raise ValueError('Trace requires finite states and monotonic timestamps')
    positive_dt = np.diff(times)[np.diff(times) > 1e-9]
    dt = float(np.median(positive_dt)) if len(positive_dt) else .04
    head_frames = max(1, int(round(keep_seconds / dt)))
    keep = np.ones(len(trace), dtype=bool)
    removed_at = np.zeros(len(trace))
    spans = []

    def key(row):
        return tuple(row.get(k) for k in ('stage', 'review_phase', 'active_object', 'look_object'))

    def protected(row):
        text = f"{row.get('stage', '')} {row.get('review_phase', '')}".lower()
        return 'human' in text or 'dynamic change' in text or 'settle physical food' in text

    def finish(start, end):
        duration = times[end] - times[start]
        if end-start <= head_frames or duration < minimum_pause or protected(trace[start]):
            return
        cut = start + head_frames
        removed = times[end] - times[cut-1] - dt
        if removed <= 0:
            return
        keep[cut:end] = False
        removed_at[end] = removed
        spans.append(dict(source_start_index=start, source_end_index=end,
                          source_start_time=float(times[start]), source_end_time=float(times[end]),
                          stage=trace[start]['stage'], removed_seconds=float(removed)))

    start = 0
    low = high = qpos[0].copy()
    for i in range(1, len(trace)):
        next_low, next_high = np.minimum(low, qpos[i]), np.maximum(high, qpos[i])
        if key(trace[i]) != key(trace[start]) or np.any(next_high-next_low > tolerance):
            finish(start, i-1)
            start = i
            low = high = qpos[i].copy()
        else:
            low, high = next_low, next_high
    finish(start, len(trace)-1)
    cumulative = np.cumsum(removed_at)
    rows = [dict(trace[i], source_index=int(i), source_time=float(times[i]),
                 time=float(times[i]-cumulative[i])) for i in np.flatnonzero(keep)]
    summary = dict(input_frames=len(trace), output_frames=len(rows),
                   original_duration_seconds=float(times[-1]-times[0]),
                   trimmed_duration_seconds=float(rows[-1]['time']-rows[0]['time']),
                   removed_seconds=float(cumulative[-1]), tolerance_qpos=tolerance,
                   minimum_pause_seconds=minimum_pause, retained_hold_seconds=keep_seconds,
                   source_timestamps_preserved=True, spans=spans,
                   scope='edited observation trajectory; raw physical rollout retained separately')
    return rows, summary


def prepare_video_trace(controller):
    if not getattr(controller, 'trim_video_pauses', False):
        return controller.trace
    rows, summary = trim_stationary_holds(controller.trace)
    output = Path(controller.output)
    (output/'trace_trimmed.json').write_text(json.dumps(rows))
    (output/'trajectory_cleanup.json').write_text(json.dumps(summary, indent=2))
    controller.report['trajectory_cleanup'] = summary
    controller.report['video_trajectory'] = str(output/'trace_trimmed.json')
    return rows
