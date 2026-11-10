"""Additional generic-object grasp configuration; vessel defaults stay fixed."""
import numpy as np


def grasp_world_capacity(category, scene_box_count, target_points):
    if category in ('cup','bowl'):
        return max(16,scene_box_count)
    # Generic objects enable cuRobo fallback during free approach. That world
    # contains both the scene and the grasp target. Merging never increases
    # voxel count, so this reserves enough space without dropping geometry.
    target_voxels=np.unique(np.floor(np.asarray(target_points)/.01).astype(int),axis=0)
    return max(16,scene_box_count+len(target_voxels))


def measured_pad_gap(geometry, joints):
    """Opposing contact-face gap from robot calibration and finger encoders.

    Joint hard limits are not the empty closing position: pads meet before
    those limits. No scene contacts or object poses enter this measurement.
    """
    tcp=geometry.tcp_pose(joints);axis=tcp[:3,1]
    intervals={name:[] for name in ('left','right')}
    for box in geometry.boxes(joints):
        for name in intervals:
            if '/'+name+'_pad' in box.name:
                center=float((box.center-tcp[:3,3])@axis)
                extent=float(np.abs(box.axes.T@axis)@box.half)
                intervals[name].append((center-extent,center+extent))
    if any(not value for value in intervals.values()):
        raise ValueError('Missing calibrated opposing finger pads')
    left,right=intervals.values()
    if np.mean(left)>np.mean(right):left,right=right,left
    return float(min(lo for lo,hi in right)-max(hi for lo,hi in left))


def observed_pick_payload(pick, geometry):
    """The same pre-grasp RGB-D estimate used by cross-room carry."""
    import json
    from pathlib import Path
    from ..runtime import artifacts
    from .observed_grasp_planner import observed_target_mask
    pick=Path(pick)
    meta=json.loads(artifacts.read_text(pick/'continuation.json'))
    report=json.loads(artifacts.read_text(pick/'report.json'))
    data=artifacts.load_array(meta['input'])
    mask=observed_target_mask(data['points'],data['target'],report['world_stats']['support_z'])
    q=artifacts.load_array(pick/'validated_paths.npz')['grasp'][-1]
    pose=geometry.tcp_pose(q)
    return (data['points'][mask]-pose[:3,3])@pose[:3,:3]
