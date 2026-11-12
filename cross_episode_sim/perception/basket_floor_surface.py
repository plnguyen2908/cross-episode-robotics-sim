"""Extract a basket floor from observed depth, avoiding point-on-rim slices.

Normals are measured before height selection: slicing a vertical wall first
can make its thin band falsely look horizontal. The caller supplies only a
visually identified basket mask, never fixture dimensions or simulator state.
"""
import numpy as np
from scipy.ndimage import label
from .surfaces import compact_surface


def basket_floor_points(camera, detection):
    if detection.mask is None:return np.empty((0,3))
    depth=np.asarray(camera.depth_m)
    height,width=depth.shape
    y,x=np.mgrid[:height,:width]
    xyz=np.stack([(x-camera.K[0,2])*depth/camera.K[0,0],
                  (y-camera.K[1,2])*depth/camera.K[1,1],depth],axis=-1)
    xyz=xyz@camera.T_base_optical[:3,:3].T+camera.T_base_optical[:3,3]
    valid=np.asarray(camera.valid_depth,bool)&np.asarray(detection.mask,bool)
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2]
    dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
    flat=np.zeros_like(valid)
    flat[1:-1,1:-1]=(abs(normal[:,:,2])>.97*length)&(length>1e-10)&(np.linalg.norm(dx,axis=-1)<.05)&(np.linalg.norm(dy,axis=-1)<.05)
    flat &= valid
    components,count=label(flat)
    planes=[]
    for index in range(1,count+1):
        points=xyz[components==index]
        if len(points)<30:continue
        centered=points-np.median(points,axis=0)
        _,singular,axes=np.linalg.svd(centered,full_matrices=False)
        if abs(axes[-1,2])<.97 or singular[1]<.025:continue
        planes.append((float(np.median(points[:,2])),points))
    if not planes:return np.empty((0,3))
    # Bottom is the lowest observed horizontal part of this basket instance.
    # Merge disconnected coplanar floor patches around existing groceries.
    bottom=min(z for z,_ in planes)
    points=np.concatenate([p for z,p in planes if abs(z-bottom)<.01])
    return compact_surface(points,resolution=.008,max_points=10000)
