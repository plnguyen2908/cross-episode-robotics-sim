"""Blend planned in-place turns into base translations for a holonomic base.

Route planners emit turn-in-place / drive-forward primitives. TidyBot++ is
holonomic, so executing them literally stops and spins the robot at every
corner, and raw arctan2 headings sometimes make it spin the long way round.
`blend_route` rewrites a planned route into segments that rotate while
translating:

- every heading change takes the shorter direction;
- moves shorter than `hop_m` (dock/grid connectors) slide at the current
  heading instead of turning to face a few centimetres of travel;
- longer moves rotate to the travel heading early in the segment, so the
  chassis (and head camera) face where the robot is going for most of it;
- the final heading is reached during the last move, not by a separate spin.

Every new segment is re-checked with the caller's collision probe at the same
25 mm / 5 degree spacing the planners use. A move whose blended sweep is not
clear falls back to turn-then-drive primitives with the shortest turns. If
that is not clear either, the function returns None and the caller executes
the planned route unchanged.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np


def wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 10 * u**3 - 15 * u**4 + 6 * u**5


@dataclass
class BaseSegment:
    """One base command: quintic x/y ramp with a staged yaw profile.

    Yaw moves from start to `cruise_yaw` over the first `turn_in` fraction of
    the segment and from `cruise_yaw` to the end yaw over the last `turn_out`
    fraction. Yaw values are continuous (not wrapped), matching the unlimited
    base hinge.
    """

    start: np.ndarray
    end: np.ndarray
    cruise_yaw: float
    turn_in: float
    turn_out: float
    duration: float
    kind: str  # "blend", "slide", "turn", "drive" or "reverse"

    def pose(self, u):
        s = smoothstep(u)
        xy = self.start[:2] + s * (self.end[:2] - self.start[:2])
        yaw = self.start[2]
        if self.turn_in > 0:
            yaw += (self.cruise_yaw - self.start[2]) * smoothstep(u / self.turn_in)
        if self.turn_out > 0:
            yaw += (self.end[2] - self.cruise_yaw) * smoothstep(
                (u - (1 - self.turn_out)) / self.turn_out
            )
        return np.array([*xy, yaw])

    @property
    def length(self):
        return float(np.linalg.norm(self.end[:2] - self.start[:2]))

    @property
    def rotation(self):
        return abs(self.cruise_yaw - self.start[2]) + abs(self.end[2] - self.cruise_yaw)

    def samples(self):
        count = max(ceil(self.length / 0.025), ceil(self.rotation / np.radians(5)), 1)
        return [self.pose(u) for u in np.linspace(0.0, 1.0, count + 1)]


def make_segment(start, end_xy, cruise_yaw, end_yaw, kind, nav_speed, turn_speed,
                 minimum=2.5):
    start = np.asarray(start, dtype=float)
    end = np.array([*end_xy, end_yaw], dtype=float)
    length = float(np.linalg.norm(end[:2] - start[:2]))
    turn_in = abs(cruise_yaw - start[2])
    turn_out = abs(end_yaw - cruise_yaw)
    duration = max(minimum, length / nav_speed, (turn_in + turn_out) / turn_speed)
    # Rotation phases take their share of the segment at the mean turn speed,
    # but never less than 30% so the heading never snaps.
    fraction_in = min(1.0, max(0.3, turn_in / turn_speed / duration)) if turn_in > 1e-3 else 0.0
    fraction_out = min(1.0, max(0.3, turn_out / turn_speed / duration)) if turn_out > 1e-3 else 0.0
    if kind == "turn":
        fraction_in, fraction_out = 0.0, 1.0
    if fraction_in + fraction_out > 1:
        total = fraction_in + fraction_out
        fraction_in, fraction_out = fraction_in / total, fraction_out / total
    return BaseSegment(start, end, float(cruise_yaw), fraction_in, fraction_out, duration, kind)


def primitive_segments(start, end_xy, drive_yaw, end_yaw, nav_speed, turn_speed):
    """Legacy turn / drive / turn for one move, as BaseSegments."""
    segments, current = [], np.asarray(start, dtype=float)
    if abs(drive_yaw - current[2]) > 1e-3:
        segments.append(make_segment(current, current[:2], current[2], drive_yaw, "turn",
                                     nav_speed, turn_speed, minimum=3.0))
        current = segments[-1].end
    if np.linalg.norm(np.asarray(end_xy) - current[:2]) > 1e-6:
        segments.append(make_segment(current, end_xy, drive_yaw, drive_yaw, "drive",
                                     nav_speed, turn_speed))
        current = segments[-1].end
    if abs(end_yaw - current[2]) > 1e-3:
        segments.append(make_segment(current, current[:2], current[2], end_yaw, "turn",
                                     nav_speed, turn_speed, minimum=3.0))
    return segments


def blend_route(path, start_yaw, pose_clear, *, nav_speed, turn_speed,
                reverse_first=False, hop_m=0.25):
    """Return BaseSegments for `path`, or None to execute the plan unchanged."""
    path = [np.asarray(p, dtype=float) for p in path]
    if len(path) < 2:
        return []
    # Continuous yaw anchored on the measured hinge angle; each planned
    # heading change is taken the short way.
    yaws = [start_yaw + wrap(path[0][2] - start_yaw)]
    for a, b in zip(path, path[1:]):
        yaws.append(yaws[-1] + wrap(b[2] - a[2]))
    poses = [np.array([*p[:2], y]) for p, y in zip(path, yaws)]
    final_yaw = poses[-1][2]
    moves = [i for i in range(len(poses) - 1)
             if np.linalg.norm(poses[i + 1][:2] - poses[i][:2]) > 1e-6]

    def clear(segments):
        return all(pose_clear(p) for s in segments for p in s.samples())

    segments, current = [], poses[0].copy()
    if reverse_first and moves and moves[0] == 0:
        reverse = make_segment(current, poses[1][:2], current[2], current[2], "reverse",
                               min(0.08, nav_speed), turn_speed)
        if not clear([reverse]):
            return None
        segments.append(reverse)
        current = reverse.end.copy()
        moves = moves[1:]
    if not moves:
        if abs(final_yaw - current[2]) > 1e-3:
            turn = make_segment(current, current[:2], current[2], final_yaw, "turn",
                                nav_speed, turn_speed, minimum=3.0)
            if not clear([turn]):
                return None
            segments.append(turn)
        return segments

    for j, i in enumerate(moves):
        target = poses[i + 1]
        last = j == len(moves) - 1
        length = float(np.linalg.norm(target[:2] - current[:2]))
        drive_yaw = current[2] + wrap(target[2] - current[2])
        sliding = length < hop_m
        cruise = current[2] if sliding else drive_yaw
        end_yaw = cruise + wrap(final_yaw - cruise) if last else cruise
        blended = make_segment(current, target[:2], cruise, end_yaw,
                               "slide" if sliding else "blend", nav_speed, turn_speed)
        if clear([blended]):
            chosen = [blended]
        else:
            chosen = primitive_segments(current, target[:2], drive_yaw,
                                        end_yaw if last else drive_yaw,
                                        nav_speed, turn_speed)
            if not clear(chosen):
                return None
        segments.extend(chosen)
        current = chosen[-1].end.copy()
    return segments
