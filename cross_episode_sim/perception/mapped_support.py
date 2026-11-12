"""Fresh measured support associated with an already labeled map landmark.

Map geometry limits association; it never fabricates depth or free space.
No new object-recognition or segmentation model is called for known furniture.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .surfaces import project_mask,table_surface_mask
from .visual_targets import Detection


def support_landmark(memory,label):
    candidates=[v for v in memory.get('furniture',[]) if v['label']==label and v.get('observations',0)>=2]
    if not candidates:
        candidates=[v for v in memory.get('tables',[]) if v['label']==label and v.get('observations',0)>=2]
    # Current protocol has one dining room. Do not silently switch instances
    # in an ambiguous map: the caller must supply a selected destination.
    return candidates[0] if len(candidates)==1 else None


def mapped_support_detection(camera,T_map_base,landmark,label):
    """Project observed bounds and fit fresh horizontal depth within them."""
    T=np.asarray(T_map_base)@camera.T_base_optical
    valid=camera.valid_depth.copy()
    if camera.robot_mask is not None:valid &= ~camera.robot_mask
    xyz=project_mask(camera.depth_m,valid,camera.K,T)
    center=np.asarray(landmark['position'])
    bounds=landmark.get('observed_bounds')
    surfaces=np.asarray(landmark.get('surface_points',[])).reshape(-1,3)
    if bounds is None and len(surfaces)>=30:
        bounds=[surfaces.min(0),surfaces.max(0)]
    if bounds is not None:
        lo,hi=np.asarray(bounds)
        nearby=((xyz[:,:2]>=lo[:2]-.08)&(xyz[:,:2]<=hi[:2]+.08)).all(1)
    else:
        # A search radius around the sensor landmark, not a claimed footprint.
        nearby=np.linalg.norm(xyz[:,:2]-center[:2],axis=1)<1.0
    nearby &= abs(xyz[:,2]-center[2])<.18
    region=np.zeros_like(valid);region[np.where(valid)]=nearby
    mask=table_surface_mask(camera.depth_m,region,camera.K,T,
        min_surface_fraction=.2,min_height=max(.35,center[2]-.18),
        max_height=min(1.5,center[2]+.18))
    if not mask.any():return None
    # Point at measured exposed plane, away from holes, the arm and boundaries.
    clearance=distance_transform_edt(mask)
    v,u=np.unravel_index(np.argmax(clearance),mask.shape)
    yy,xx=np.where(mask)
    return Detection(label,1.,(float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)),mask,(int(u),int(v)))
