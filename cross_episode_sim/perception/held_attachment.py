"""Refresh attachment extent from a semantically pointed, onboard RGB-D mask.

This is local manipulation perception, not a transit check. Prior geometry is
retained so a partial view cannot make the carried object smaller.
"""
import numpy as np
from scipy.spatial import cKDTree

from .point_masks import PointMaskDetector


def connected_attachment_surface(points, prior):
    """Keep connected mask components anchored to the prior held surface.

    Segmentation pixels at depth discontinuities can unproject onto a distant
    background. Split actual 3-D components first; never cut a connected
    counter surface down to fit the expected object.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points = np.asarray(points, float)
    if len(points) < 30:
        return np.empty((0, 3)), []
    voxels, inverse = np.unique(np.floor(points/.006).astype(int), axis=0, return_inverse=True)
    centers = (voxels+.5)*.006
    pairs = cKDTree(centers).query_pairs(.0125, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(centers), len(centers)))
    _, labels = connected_components(graph, directed=False)
    tree = cKDTree(prior)
    accepted, evidence = [], []
    for label in np.unique(labels):
        component = points[labels[inverse] == label]
        if len(component) < 30:
            continue
        distances = tree.query(component)[0]
        extent = float(np.linalg.norm(np.ptp(prior, axis=0)))
        anchored = np.count_nonzero(distances < .015) >= 30
        bounded = np.linalg.norm(np.ptp(component, axis=0)) <= 1.5*extent
        valid = bool(anchored and bounded
                     and np.median(distances) < max(.025, .15*extent)
                     and np.max(distances) < max(.08, .4*extent))
        evidence.append(dict(points=len(component), accepted=valid,
                             median_distance_m=float(np.median(distances)),
                             max_distance_m=float(distances.max())))
        if valid:
            accepted.append(component)
    return (np.concatenate(accepted) if accepted else np.empty((0, 3))), evidence


def observe_held_attachment(packet, geometry, prior, category):
    detector = PointMaskDetector(categories=(f'{category} held by the robot gripper',))
    tcp = geometry.tcp_pose(packet.arm.position)
    prior = np.asarray(prior, float)
    accepted, evidence = [], []
    for name, camera in packet.cameras.items():
        for detection in detector.detect(camera.rgb):
            if detection.mask is None:
                continue
            mask = detection.mask & camera.valid_depth
            if camera.robot_mask is not None:
                mask &= ~camera.robot_mask
            v, u = np.where(mask)
            z = camera.depth_m[v, u]
            optical = np.c_[(u-camera.K[0, 2])*z/camera.K[0, 0],
                            (v-camera.K[1, 2])*z/camera.K[1, 1], z]
            points = optical @ camera.T_base_optical[:3, :3].T + camera.T_base_optical[:3, 3]
            for box in geometry.boxes(packet.arm.position):
                points = points[~np.all(np.abs((points-box.center)@box.axes) <= box.half+.01, axis=1)]
            tool = (points-tcp[:3, 3]) @ tcp[:3, :3]
            if len(tool) < 30:
                continue
            matched, components = connected_attachment_surface(tool, prior)
            evidence.append(dict(camera=name, points=len(tool), components=components))
            if len(matched):
                accepted.append(matched)
    if not accepted:
        raise ValueError(f'No semantically associated held surface: {evidence}')
    combined = np.concatenate([prior, *accepted])
    _, indices = np.unique(np.floor(combined/.005).astype(int), axis=0, return_index=True)
    return combined[indices], evidence
