"""Plan a forward object view from estimated pose and own camera calibration."""
import math
import numpy as np
from scipy.ndimage import binary_dilation


def visible_object_heading(base_pose,target,camera,xy=None):
    """Return the closest forward-facing heading with no robot self-occlusion.

    The arm must remain in the checked stowed posture. Its onboard silhouette
    is then rigid relative to the mast camera. Environment occlusion still
    requires fresh RGB-D confirmation after arrival; this is not a grasp pose.
    """
    point=np.asarray(target);pose=np.asarray(base_pose).copy()
    if xy is not None:pose[:2,3]=xy
    bearing=math.atan2(point[1]-pose[1,3],point[0]-pose[0,3])
    old_yaw=math.atan2(pose[1,0],pose[0,0])
    h,w=camera.depth_m.shape;margin=max(4,int(min(h,w)*.03))
    robot=np.zeros((h,w),bool) if camera.robot_mask is None else binary_dilation(camera.robot_mask,iterations=margin)
    for offset in sorted(np.linspace(-.75,.75,31),key=abs):
        heading=bearing+offset;change=heading-old_yaw;c,s=math.cos(change),math.sin(change)
        pose[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])@base_pose[:3,:3]
        T=np.linalg.inv(pose@camera.T_base_optical)
        q=T[:3,:3]@point+T[:3,3]
        if q[2]<=.1:continue
        pixel=camera.K@q;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
        if margin<=u<w-margin and margin<=v<h-margin and not robot[v,u]:
            return float(math.atan2(math.sin(heading),math.cos(heading)))
    return None
