"""Match learned contact depth to the calibrated gripper's pad length.

A proposed contact beyond the fingertips can pass arm collision checks yet
close empty. This filter uses only the learned observed-surface contact and
robot calibration. It does not establish force closure or physical success;
the executor still validates paths and measured grip retention.
"""
import numpy as np


def pad_depth_interval(geometry, joints, margin=.003):
    tcp = geometry.tcp_pose(joints)
    bounds = []
    for box in geometry.boxes(joints):
        if '/left_pad' not in box.name and '/right_pad' not in box.name:
            continue
        center = (box.center-tcp[:3, 3]) @ tcp[:3, :3]
        extent = np.abs(tcp[:3, :3].T @ box.axes) @ box.half
        bounds.append((center[2]-extent[2], center[2]+extent[2]))
    if not bounds:
        raise ValueError('Missing calibrated opposing finger pads')
    low = min(b[0] for b in bounds) + margin
    high = max(b[1] for b in bounds) - margin
    if low >= high:
        raise ValueError('Invalid calibrated pad depth interval')
    return low, high


def contact_depth_matches(row, interval):
    tcp = np.asarray(row['tcp'], float)
    contact = (np.asarray(row['contact'])-tcp[:3, 3]) @ tcp[:3, :3]
    return bool(interval[0] <= contact[2] <= interval[1])


def calibrated_contact_variants(rows, geometry, joints):
    interval = pad_depth_interval(geometry, joints)
    for row in rows:
        for backoff in (0., -.01, -.02, .005, .01):
            pose = np.asarray(row['tcp'], float).copy()
            pose[:3, 3] -= backoff*pose[:3, 2]
            candidate = dict(row, tcp=pose.tolist(), contact_backoff=backoff,
                             calibrated_pad_depth_interval=list(interval))
            if contact_depth_matches(candidate, interval):
                yield candidate
