"""Restrict a semantic object mask to the metric region around a part point."""
import numpy as np
from .visual_targets import Detection


def pointed_region(camera,detection,radius=.16):
    if detection.point is None:return None
    u,v=map(int,detection.point);valid=camera.valid_depth.copy()
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    if not (0<=v<valid.shape[0] and 0<=u<valid.shape[1]) or not valid[v,u]:return None
    yy,xx=np.indices(valid.shape);z=camera.depth_m
    local=np.stack([(xx-camera.K[0,2])*z/camera.K[0,0],(yy-camera.K[1,2])*z/camera.K[1,1],z],axis=-1)
    xyz=local@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
    seed=xyz[v,u]
    mask=valid&(np.linalg.norm(xyz[...,:2]-seed[:2],axis=-1)<=radius)&(abs(xyz[...,2]-seed[2])<.08)
    if detection.mask is not None:mask &= detection.mask
    rows,cols=np.where(mask)
    if len(rows)<20:return None
    return Detection(detection.label,detection.score,(int(cols.min()),int(rows.min()),int(cols.max())+1,int(rows.max())+1),mask,detection.point)
